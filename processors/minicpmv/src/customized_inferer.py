# Customized deep learning model inferer. Including model load and model infer.
import gc
import json
import os
import sys
from copy import deepcopy

from PIL import Image
from transformers import AutoModel, AutoProcessor

# Add src dir to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

from grps_framework.context.context import GrpsContext
from grps_framework.logger.logger import clogger
from grps_framework.model_infer.inferer import ModelInferer, inferer_register


class YourInferer(ModelInferer):
    def __init__(self):
        super(YourInferer, self).__init__()
        self.hf_model = None
        self.dtype = None
        self.processor = None
        self.vision_batch_size = None

    def init(self, path, device=None, args=None):
        """
        Initiate model inferer class with model path and device.

        Args:
            path: Model path, it can be a file path or a directory path.
            device: Device to run model.
            args: More args.

        Raises:
            Exception: If init failed, can raise exception. Will be caught by server and show error message to user when
            start service.
        """
        super(YourInferer, self).init(path, device, args)
        dtype = args.get('dtype', 'bfloat16')
        if dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        elif dtype == 'float16':
            self.dtype = torch.float16
        else:
            raise ValueError('dtype should be bfloat16 or float16.')
        self.vision_batch_size = args.get('vision_batch_size', 6)
        clogger.info('your inferer init, path: {}, device: {}, dtype: {}'.format(path, device, self.dtype))

    def load(self):
        """
        Load model from model path.

        Returns:
            True if load model successfully, otherwise False.

        Raises:
            Exception: If load failed, can raise exception and exception will be caught by server and show error message
            to user when start service.
        """
        clogger.info('your inferer load...')
        self.hf_model = AutoModel.from_pretrained(self._path, trust_remote_code=True,
                                                  attn_implementation='sdpa',
                                                  device_map='cpu')  # sdpa or flash_attention_2, no eager
        self.hf_model.eval()
        self.processor = AutoProcessor.from_pretrained(self._path, trust_remote_code=True)

        # free llm memory, only use vit.
        del self.hf_model.llm
        gc.collect()
        self.hf_model.vpm.to(self._device).to(self.dtype)
        self.hf_model.resampler.to(self._device).to(self.dtype)
        clogger.info('your inferer load model successfully.')
        return True

    def infer(self, inp, context: GrpsContext):
        """
        The inference function is used to make a prediction call on the given input request.

        Args:
            context: grps context
            inp: Model infer input, which is output of converter preprocess function. When in `no converter mode`, will
            skip converter preprocess and directly use GrpsMessage as input.

        Returns:
            Model infer output, which will be input of converter postprocess function. When in `no converter mode`, it
            will skip converter postprocess and should directly use GrpsMessage as output.

        Raises:
            Exception: If infer failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        msgs_list = [inp]
        images_list = [None] * len(msgs_list)

        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            if len(msgs) <= 0:
                raise ValueError("The input message list is empty.")

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            system_prompt = None
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                if role in ["user", "assistant"]:
                    if isinstance(content, str):
                        content = [content]
                    cur_msgs = []
                    for c in content:
                        if isinstance(c, Image.Image):
                            images.append(c)
                            cur_msgs.append("(<image>./</image>)")
                        elif isinstance(c, str):
                            cur_msgs.append(c)
                    msg["content"] = "\n".join(cur_msgs)
                elif role == 'system':
                    system_prompt = msg["content"]
                else:
                    raise ValueError(f"Invalid role: {role}")

            if system_prompt:
                sys_msg = {'role': 'system', 'content': system_prompt}
                copy_msgs = [sys_msg] + copy_msgs

            prompts_lists.append(
                self.processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            # clogger.info('final prompts_lists: {}'.format(prompts_lists))
            input_images_lists.append(images)

        inputs = self.processor(
            prompts_lists,
            input_images_lists,
            max_slice_nums=None,
            use_image_id=None,
            return_tensors="pt",
            max_length=None
        ).to(self._device)
        # clogger.info('final inputs: {}'.format(inputs))

        if len(inputs['tgt_sizes'][0]) == 0:
            return inputs["input_ids"], None
        with torch.inference_mode():
            inputs.pop("image_sizes")
            tgt_sizes = inputs['tgt_sizes']
            pixel_values_list = inputs['pixel_values']
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
            tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)
            max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                               padding_value=0.0)
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=self._device)
            for i in range(B):
                patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

            all_pixel_values = all_pixel_values.type(self.dtype)
            if B > self.vision_batch_size:
                hs = []
                for i in range(0, B, self.vision_batch_size):
                    start_idx = i
                    end_idx = i + self.vision_batch_size
                    tmp_hs = self.hf_model.vpm(
                        all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                        tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                    hs.append(tmp_hs)
                vision_embedding = torch.cat(hs, dim=0)
            else:
                vision_embedding = self.hf_model.vpm(all_pixel_values, patch_attention_mask=patch_attn_mask,
                                                     tgt_sizes=tgt_sizes).last_hidden_state
            vision_embedding = self.hf_model.resampler(vision_embedding, tgt_sizes)
            return inputs["input_ids"], vision_embedding

    def batch_infer(self, inp, contexts: list):
        pass


# Register
inferer_register.register('your_inferer', YourInferer())
