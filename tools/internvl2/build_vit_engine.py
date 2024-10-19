import argparse
import os
from time import time

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision.transforms as T
from PIL import Image
from tensorrt_llm._utils import str_dtype_to_torch
from torchvision.transforms import InterpolationMode
from transformers import AutoModelForCausalLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    print('pixel_values shape: ', pixel_values.shape)
    return pixel_values


class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, vision_model, vision_mlp1, select_layer=-1):
        super().__init__()
        self.vision_model = vision_model
        self.mlp1 = vision_mlp1
        self.downsample_ratio = 0.5
        self.select_layer = select_layer

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        # if self.ps_version == 'v1':
        #     warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
        #                   'which results in a transposed image.')
        # else:
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds


class ONNX_TRT:

    def __init__(self, image_size=448):
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
                            opt_bs=13,
                            max_bs=4 * 13):
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


def trt_malloc(inp_data):
    """malloc tensorrt input and output cpu and gpu memory."""
    h_input = np.array(inp_data)
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(
        4 * 13 * 3 * 448 * 448 * 2)  # can allocate a larger batch size to reuse of this memory
    d_output = cuda.mem_alloc(4 * 13 * 3 * 256 * 4096)  # can allocate a larger batch size to reuse of this memory
    return h_input, d_input, d_output


def trt_infer(inp_name, out_name, h_input, d_input, d_output, trt_ctx, stream):
    """pure-trt infer."""
    batch_size = h_input.shape[0]
    # set true input shape
    print('h_input.shape: ', h_input.shape)
    trt_ctx.set_input_shape(inp_name, h_input.shape)
    # copy input data from cpu to gpu
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # execute trt engine
    trt_ctx.set_tensor_address(inp_name, int(d_input))
    trt_ctx.set_tensor_address(out_name, int(d_output))
    trt_ctx.execute_async_v3(stream_handle=stream.handle)
    # copy output data from gpu to cpu
    h_output = cuda.pagelocked_empty((batch_size, 256, 4096), dtype=np.float16)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # synchronize stream
    stream.synchronize()
    print('h_output.shape: ', h_output.shape)
    return h_output


def compare_output(vision_encoder, trt_file, image):
    print("Start comparing the output of HF model and TRT engine!")
    # HF infer
    hf_output = vision_encoder(image)
    print("HF output: ", hf_output)

    # TRT infer
    import pycuda.autoinit

    trt_engine, trt_ctx, inp_name, out_name = load_trt_engine(trt_file)
    h_input, d_input, d_output = trt_malloc(np.ascontiguousarray(image.numpy()))
    stream = cuda.Stream()
    trt_out = trt_infer(inp_name, out_name, h_input, d_input, d_output, trt_ctx, stream)
    print("TRT output: ", trt_out)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # onnx/vision_encoder
    parser.add_argument('--imagePath',
                        type=str,
                        default='./data/image1.jpg',
                        help='')
    parser.add_argument('--onnxFile',
                        type=str,
                        default='/tmp/InternVL2-8B/vision_encoder_bfp16.onnx',
                        help='')
    parser.add_argument('--pretrainedModelPath',
                        type=str,
                        default='/tmp/InternVL2-8B',
                        help='')
    parser.add_argument('--trtFile',
                        type=str,
                        default='/tmp/InternVL2-8B/vision_encoder_bfp16.trt',
                        help='')
    parser.add_argument('--onlyTrt',
                        action='store_true',
                        help='Run only convert the onnx to TRT engine.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        help='The dtype of the model.',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--minBS', type=int, default=1)
    parser.add_argument('--optBS', type=int, default=13)
    parser.add_argument('--maxBS', type=int, default=4 * 13)
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

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.pretrainedModelPath,
        device_map='cpu',
        torch_dtype=str_dtype_to_torch(args.dtype),
        trust_remote_code=True,
    ).eval()
    vision_encoder = (VisionEncoderWrapper(hf_model.vision_model, hf_model.mlp1, select_layer=-1).to('cpu'))

    image = load_image(args.imagePath).to('cpu').to(str_dtype_to_torch(args.dtype))

    onnx_trt_obj = ONNX_TRT()

    if args.onlyTrt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.trtFile,
                                         args.minBS, args.optBS, args.maxBS)
    else:
        onnx_trt_obj.export_onnx(image, args.onnxFile, vision_encoder)
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.trtFile,
                                         args.minBS, args.optBS, args.maxBS)

    # compare_output(vision_encoder, args.trtFile, image)
