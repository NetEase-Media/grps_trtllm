import argparse
import os
from time import time

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
from tensorrt_llm._utils import str_dtype_to_torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, vision_tower, multi_modal_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(self, pixel_values):
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features


class ONNX_TRT:

    def __init__(self, image_size=896):
        self.image_size = image_size

    def export_onnx(self, image, onnx_file_path, vision_encoder):
        print("Start converting ONNX model!")

        torch.onnx.export(vision_encoder,
                          args=image,
                          f=onnx_file_path,
                          opset_version=17,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {
                              0: 'batch'
                          }})
        print("Succeeded converting ONNX model!")

    def generate_trt_engine(self,
                            onnx_file,
                            trt_file,
                            min_bs=1,
                            opt_bs=1,
                            max_bs=32):
        print("Start converting TRT engine!")

        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.BF16)
        config.set_flag(trt.BuilderFlag.FP16)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_file, 'rb') as model:
            if not parser.parse(model.read(), "/".join(onnx_file.split("/"))):
                print("Failed parsing %s" % onnx_file)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnx_file)

        input = network.get_input(0)
        input.shape = [-1, 3, self.image_size, self.image_size]
        profile.set_shape(input.name,
                          [min_bs, 3, self.image_size, self.image_size],
                          [opt_bs, 3, self.image_size, self.image_size],
                          [max_bs, 3, self.image_size, self.image_size])

        config.add_optimization_profile(profile)

        t0 = time()
        engine_str = builder.build_serialized_network(network, config)
        t1 = time()
        if engine_str is None:
            print("Failed building %s" % trt_file)
        else:
            print("Succeeded building %s in %d s" % (trt_file, t1 - t0))
        print("plan file is", trt_file)
        with open(trt_file, 'wb') as f:
            f.write(engine_str)


def load_trt_engine(engine_path):
    """load tensorrt engine and create execution context."""
    print('Loading tensorrt engine...')
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()  # create execution context
    inp_name = None
    out_name = None
    for i in range(engine.num_io_tensors):
        print("----tensor {}".format(i))
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print('name: {}'.format(name))
        print('shape: {}'.format(shape))
        print('dtype: {}'.format(dtype))
        print('vec_dim: {}'.format(engine.get_tensor_vectorized_dim(name)))
        print('comps: {}'.format(engine.get_tensor_components_per_element(name)))
        print('is_shape: {}'.format(engine.is_shape_inference_io(name)))
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        print('is_input: {}'.format(is_input))
        if is_input:
            print('get_profile_shape: {}'.format(engine.get_tensor_profile_shape(name, 0)))
            inp_name = name
        else:
            out_name = name

    return engine, context, inp_name, out_name


def trt_infer(inp_name, out_name, trt_ctx, stream, inp_data):
    """pure-trt infer."""
    """malloc tensorrt input and output cpu and gpu memory."""
    h_input = np.array(inp_data.cpu().numpy())
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)

    # set true input shape
    print('h_input.shape: ', h_input.shape)
    trt_ctx.set_input_shape(inp_name, h_input.shape)

    # get true output shape.
    d_out_shape = trt_ctx.get_tensor_shape(out_name)
    print('d_output.shape: ', d_out_shape)
    d_output = cuda.mem_alloc(trt.volume(d_out_shape) * 2)

    # copy input data from cpu to gpu
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # execute trt engine
    trt_ctx.set_tensor_address(inp_name, int(d_input))
    trt_ctx.set_tensor_address(out_name, int(d_output))
    trt_ctx.execute_async_v3(stream_handle=stream.handle)
    # copy output data from gpu to cpu
    h_output = cuda.pagelocked_empty(list(d_out_shape), dtype=np.float16)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # synchronize stream
    stream.synchronize()
    # print('h_output.shape: ', h_output.shape)
    return h_output


def compare_output(vision_encoder, trt_file, image):
    print("Start comparing the output of HF model and TRT engine!")
    # HF infer
    hf_output = vision_encoder(image).cpu().detach().numpy()
    print("HF output: ", hf_output)

    # TRT infer
    import pycuda.autoinit

    trt_engine, trt_ctx, inp_name, out_name = load_trt_engine(trt_file)
    stream = cuda.Stream()
    trt_out = trt_infer(inp_name, out_name, trt_ctx, stream, image)

    print("TRT output: ", trt_out)

    # Calc diff rate
    hf_output = hf_output.astype(np.float32)
    trt_out = trt_out.astype(np.float32)
    diff = np.abs(hf_output - trt_out)
    diff_sum = np.sum(diff)
    origin_sum = np.sum(np.abs(hf_output))
    print("Diff rate: {:.2f}%， diff sum: {}, origin_sum: {}".format(diff_sum / origin_sum * 100, diff_sum, origin_sum))


def parse_arguments():
    parser = argparse.ArgumentParser()
    # onnx/vision_encoder
    parser.add_argument('--imageUrl',
                        type=str,
                        default='https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg',
                        help='')
    parser.add_argument('--onnxFile',
                        type=str,
                        default='',
                        required=True,
                        help='onnx file path')
    parser.add_argument('--pretrainedModelPath',
                        type=str,
                        default='',
                        required=True,
                        help='huggingface pretrained model path')
    parser.add_argument('--trtFile',
                        type=str,
                        default='',
                        required=True,
                        help='trt file path')
    parser.add_argument('--onlyTrt',
                        action='store_true',
                        help='Run only convert the onnx to TRT engine.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        help='The dtype of the model.',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--loadOnCpu',
                        action='store_true',
                        help='Load the model on CPU.')
    parser.add_argument('--minBS', type=int, default=1)
    parser.add_argument('--optBS', type=int, default=1)
    parser.add_argument('--maxBS', type=int, default=32)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    onnx_file_dir = os.path.dirname(args.onnxFile)
    if not onnx_file_dir == '' and not os.path.exists(onnx_file_dir):
        os.makedirs(onnx_file_dir)
    trt_file_dir = os.path.dirname(args.trtFile)
    if not os.path.exists(trt_file_dir):
        os.makedirs(trt_file_dir)

    # Load vit.
    hf_model = Gemma3ForConditionalGeneration.from_pretrained(
        args.pretrainedModelPath, device_map="cpu",
        torch_dtype=str_dtype_to_torch(args.dtype),
        trust_remote_code=True
    ).eval()
    vision_encoder = VisionEncoderWrapper(hf_model.vision_tower, hf_model.multi_modal_projector)

    # Create pixel_values.
    processor = AutoProcessor.from_pretrained(args.pretrainedModelPath)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.imageUrl},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    image = inputs['pixel_values'].to(str_dtype_to_torch(args.dtype))

    if not args.loadOnCpu:
        vision_encoder = vision_encoder.to('cuda')
        image = image.to('cuda')

    onnx_trt_obj = ONNX_TRT()

    if args.onlyTrt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.trtFile,
                                         args.minBS, args.optBS, args.maxBS)
    else:
        onnx_trt_obj.export_onnx(image, args.onnxFile, vision_encoder)
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.trtFile,
                                         args.minBS, args.optBS, args.maxBS)

    if args.dtype == 'float16':  # Current numpy not support bfloat16. Only compare float16.
        compare_output(vision_encoder, args.trtFile, image)
