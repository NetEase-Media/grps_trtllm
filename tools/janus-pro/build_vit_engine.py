import argparse
import os

import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from einops import rearrange

from tensorrt_llm._utils import str_dtype_to_torch


class VisionEncoderWrapper(torch.nn.Module):
    def __init__(self, vision_model, aligner):
        super().__init__()
        self.vision_model = vision_model
        self.aligner = aligner

    def forward(self, images):
        return self.aligner(self.vision_model(images))


class ONNX_TRT:

    def __init__(self):
        self.image_size = None

    def export_onnx(self, onnx_file_path, pretrained_model_path, image_url, load_on_cpu, dtype):
        print("Start converting ONNX model!")
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(pretrained_model_path)
        self.image_size = vl_chat_processor.image_processor.image_size

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path, trust_remote_code=True, device_map="cpu")
        device = torch.device("cuda") if not load_on_cpu else "cpu"
        vision_model = VisionEncoderWrapper(vl_gpt.vision_model, vl_gpt.aligner)
        vision_model = vision_model.to(str_dtype_to_torch(dtype)).to(device).eval()

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n这是什么？",
                "images": [image_url],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(device)
        images = rearrange(prepare_inputs.pixel_values, "b n c h w -> (b n) c h w").to(str_dtype_to_torch(dtype))

        print('images shape:', images.shape)
        torch.onnx.export(vision_model,
                          images,
                          onnx_file_path,
                          opset_version=17,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {
                              0: 'batch'
                          }})

    def generate_trt_engine(self,
                            onnxFile,
                            planFile,
                            minBS=1,
                            optBS=2,
                            maxBS=4):
        print("Start converting TRT engine!")
        from time import time

        import tensorrt as trt
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.BF16)
        config.set_flag(trt.BuilderFlag.FP16)
        parser = trt.OnnxParser(network, logger)

        with open(onnxFile, 'rb') as model:
            if not parser.parse(model.read(), "/".join(onnxFile.split("/"))):
                print("Failed parsing %s" % onnxFile)
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
            print("Succeeded parsing %s" % onnxFile)

        input = network.get_input(0)
        input.shape = [-1, 3, self.image_size, self.image_size]
        profile.set_shape(input.name,
                          [minBS, 3, self.image_size, self.image_size],
                          [optBS, 3, self.image_size, self.image_size],
                          [maxBS, 3, self.image_size, self.image_size])

        config.add_optimization_profile(profile)

        t0 = time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time()
        if engineString == None:
            print("Failed building %s" % planFile)
        else:
            print("Succeeded building %s in %d s" % (planFile, t1 - t0))
        print("plan file is", planFile)
        with open(planFile, 'wb') as f:
            f.write(engineString)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # onnx/visual_encoder
    parser.add_argument('--onnxFile',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--pretrainedModelPath',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--trtFile',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--onlyTrt',
                        action='store_true',
                        help='Run only convert the onnx to TRT engine.')
    parser.add_argument('--loadOnCpu',
                        action='store_true',
                        help='Load the model on CPU.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        help='The dtype of the model.',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--minBS', type=int, default=1)
    parser.add_argument('--optBS', type=int, default=1)
    parser.add_argument('--maxBS', type=int, default=4)
    parser.add_argument('--imagePath', type=str, default='./images/logo.png')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    onnx_file_dir = os.path.dirname(args.onnxFile)
    if not onnx_file_dir == '' and not os.path.exists(onnx_file_dir):
        os.makedirs(onnx_file_dir)
    plan_file_dir = os.path.dirname(args.trtFile)
    if not os.path.exists(plan_file_dir):
        os.makedirs(plan_file_dir)

    onnx_trt_obj = ONNX_TRT()

    if args.onlyTrt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.trtFile,
                                         args.minBS, args.optBS, args.maxBS)
    else:
        onnx_trt_obj.export_onnx(args.onnxFile, args.pretrainedModelPath,
                                 args.imagePath, args.loadOnCpu, args.dtype)
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.trtFile,
                                         args.minBS, args.optBS, args.maxBS)
