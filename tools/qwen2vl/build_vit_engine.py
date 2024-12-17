import argparse
import math
import os
import shutil
import time
from typing import Optional, Tuple

import numpy as np
import onnx
import pycuda.driver as cuda
import requests
import tensorrt as trt
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from tensorrt_llm._utils import str_dtype_to_torch, torch_dtype_to_str
from tensorrt_llm.builder import Builder
from torch import nn
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel, Qwen2VLVisionBlock,
    VisionAttention, VisionRotaryEmbedding, apply_rotary_pos_emb_vision)


def get_rope_index(
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    spatial_merge_size = 2
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if image_grid_thw is not None or video_grid_thw is not None:
        total_input_ids = input_ids
        position_ids = torch.ones(3,
                                  input_ids.shape[0],
                                  input_ids.shape[1],
                                  dtype=input_ids.dtype,
                                  device=input_ids.device)
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                    llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) +
                    st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                    -1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                    llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                    llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len +
                    st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                    llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) +
                    st_idx)

            llm_positions = torch.cat(llm_pos_ids_list,
                                      dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 -
                                         len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[
                -1]
        else:
            position_ids = (torch.arange(input_ids.shape[1],
                                         device=input_ids.device).view(
                1, 1, -1).expand(
                3, input_ids.shape[0], -1))
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def compute_rotary_pos_emb(grid_thw, hf_config):
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // hf_config.vision_config.spatial_merge_size,
            hf_config.vision_config.spatial_merge_size,
            w // hf_config.vision_config.spatial_merge_size,
            hf_config.vision_config.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // hf_config.vision_config.spatial_merge_size,
            hf_config.vision_config.spatial_merge_size,
            w // hf_config.vision_config.spatial_merge_size,
            hf_config.vision_config.spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(
            torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    pos_ids = torch.cat(pos_ids, dim=0)
    return pos_ids


class VisionAttentionOpt(VisionAttention):

    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__(dim, num_heads)
        self.head_dim = dim / num_heads

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                  self.num_heads,
                                                  -1).permute(1, 0, 2,
                                                              3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2VLVisionBlockOpt(Qwen2VLVisionBlock):

    def __init__(self, config, attn_implementation: str = "eager") -> None:
        super().__init__(config)
        self.attn = VisionAttentionOpt(config.embed_dim,
                                       num_heads=config.num_heads)

    def forward(self, hidden_states, attention_mask,
                rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VisionTransformerPretrainedModelOpt(
    Qwen2VisionTransformerPretrainedModel):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList([
            Qwen2VLVisionBlockOpt(config, config._attn_implementation)
            for _ in range(config.depth)
        ])

    def forward(self, hidden_states: torch.Tensor,
                rotary_pos_emb: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        for blk in self.blocks:
            hidden_states = blk(hidden_states,
                                attention_mask=attention_mask,
                                rotary_pos_emb=rotary_pos_emb)
        res = self.merger(hidden_states)
        return res


class VisionEncoderWrapper(torch.nn.Module):

    def __init__(self, model, hf_config, dtype=torch.float32, rotary_pos_emb_func=None):
        super().__init__()
        self.visual = Qwen2VisionTransformerPretrainedModelOpt._from_config(
            model.config.vision_config,
            torch_dtype=dtype,
        )
        self.visual.load_state_dict(model.visual.state_dict())
        self.hf_config = hf_config
        self.rotary_pos_emb_func = rotary_pos_emb_func
        self.out_dtype = dtype

    def create_sinusoidal_positions_for_attention_plugin(
            self,
            num_pos: int,
            dim: int,
            theta: float = 10000.0,
            scale: float = 1.0,
            dtype=torch.float32):
        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = scale / (theta ** (torch.arange(0, dim, 2) / dim)).to(dtype)

        # Multiply each theta by the position (which is the argument of the sin and cos functions)
        sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=dtype), inv_freq)
        sinusoid_inp = sinusoid_inp.unsqueeze(-1)
        concat = torch.cat([torch.cos(sinusoid_inp), torch.sin(sinusoid_inp)], dim=-1)

        return inv_freq, concat.reshape(1, -1).to(dtype)

    def forward(self, images, rotary_pos_ids, image_grid_thw, attention_mask_vit, mrope_position_ids_padding,
                mrope_position_deltas):
        max_grid_size = image_grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb_func(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[rotary_pos_ids].flatten(1)

        img_features = self.visual(images, rotary_pos_emb, attention_mask_vit)
        img_features = img_features.to(self.out_dtype)

        max_position_embeddings = int(self.hf_config.max_position_embeddings)
        rotary_embedding_dim = int(self.hf_config.hidden_size / self.hf_config.num_attention_heads)
        rotary_embedding_base = float(self.hf_config.rope_theta)
        rotary_embedding_scale = float(1.0)
        inv_freq, rotary_cos_sin = self.create_sinusoidal_positions_for_attention_plugin(
            max_position_embeddings, rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale)
        rotary_cos_sin = rotary_cos_sin.reshape(max_position_embeddings, int(rotary_embedding_dim / 2), 2)
        cos_ori = rotary_cos_sin[:, :, 0]
        sin_ori = rotary_cos_sin[:, :, 1]
        cos = cos_ori[mrope_position_ids_padding]
        sin = sin_ori[mrope_position_ids_padding]

        mrope_section = [16, 24, 24]
        unsqueeze_dim = -1
        cos = torch.cat([m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim)
        sin = torch.cat([m[:, i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim)
        concat_cos_sin = torch.cat((cos, sin), dim=-1)
        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)

        return img_features, concat_cos_sin, mrope_position_deltas


class MropeOnlyWrapper(torch.nn.Module):

    def __init__(self, hf_config):
        super().__init__()
        self.hf_config = hf_config

    def create_sinusoidal_positions_for_attention_plugin(
            self,
            num_pos: int,
            dim: int,
            theta: float = 10000.0,
            scale: float = 1.0,
            dtype=torch.float32):
        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, ..., dim // 2
        inv_freq = scale / (theta ** (torch.arange(0, dim, 2) / dim)).to(dtype)

        # Multiply each theta by the position (which is the argument of the sin and cos functions)
        sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=dtype), inv_freq)
        sinusoid_inp = sinusoid_inp.unsqueeze(-1)
        concat = torch.cat([torch.cos(sinusoid_inp), torch.sin(sinusoid_inp)], dim=-1)

        return inv_freq, concat.reshape(1, -1).to(dtype)

    def forward(self, mrope_position_ids_padding, mrope_position_deltas):
        max_position_embeddings = int(self.hf_config.max_position_embeddings)
        rotary_embedding_dim = int(self.hf_config.hidden_size / self.hf_config.num_attention_heads)
        rotary_embedding_base = float(self.hf_config.rope_theta)
        rotary_embedding_scale = float(1.0)
        inv_freq, rotary_cos_sin = self.create_sinusoidal_positions_for_attention_plugin(
            max_position_embeddings, rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale)
        rotary_cos_sin = rotary_cos_sin.reshape(max_position_embeddings, int(rotary_embedding_dim / 2), 2)
        cos_ori = rotary_cos_sin[:, :, 0]
        sin_ori = rotary_cos_sin[:, :, 1]
        cos = cos_ori[mrope_position_ids_padding]
        sin = sin_ori[mrope_position_ids_padding]

        mrope_section = [16, 24, 24]
        unsqueeze_dim = -1
        cos = torch.cat([m[:, i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim)
        sin = torch.cat([m[:, i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim)
        concat_cos_sin = torch.cat((cos, sin), dim=-1)
        concat_cos_sin = concat_cos_sin.reshape(concat_cos_sin.shape[0], -1)

        return concat_cos_sin, mrope_position_deltas


def export_onnx(model,
                input,
                onnx_path='model.onnx',
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {
                    0: 'batch'
                }},
                logger=trt.Logger(trt.Logger.INFO)):
    logger.log(trt.Logger.INFO, f"Exporting onnx to {onnx_path}")

    torch.onnx.export(model,
                      input,
                      f'{onnx_path}',
                      opset_version=17,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)

    onnx.checker.check_model(onnx_path)

    logger.log(trt.Logger.INFO, f"Exported onnx to {onnx_path} successfully.")


def build_trt_engine(onnx_file,
                     engine_path,
                     dtype=torch.float16,
                     qwen2_vl_dim=0,
                     min_batch_size=1,
                     opt_batch_size=1,
                     max_batch_size=1,
                     min_hw_dims=0,
                     opt_hw_dims=0,
                     max_hw_dims=0,
                     num_frames=None,
                     delete_onnx=False,
                     logger=trt.Logger(trt.Logger.VERBOSE)):
    logger.log(trt.Logger.INFO, f"Building TRT engine to {engine_path}")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config_args = {
        "precision": torch.float32,
        "strongly_typed": False
    }
    if num_frames is not None:
        config_args["num_frames"] = num_frames

    config_wrapper = Builder().create_builder_config(**config_args)
    config = config_wrapper.trt_builder_config
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    input_images = network.get_input(0)
    pos_ids = network.get_input(1)
    image_grid_thw = network.get_input(2)
    attenstion_mask_vit = network.get_input(3)
    mrope_position_ids_padding = network.get_input(4)
    mrope_position_deltas = network.get_input(5)
    assert min_hw_dims > 0
    assert opt_hw_dims > 0
    assert max_hw_dims > 0
    assert min_batch_size > 0
    assert opt_batch_size > 0
    assert max_batch_size > 0
    multi_size_min = min_hw_dims * min_batch_size
    multi_size_opt = opt_hw_dims * opt_batch_size
    multi_size_max = max_hw_dims * max_batch_size

    input_images.shape = [-1, qwen2_vl_dim]
    profile.set_shape(input_images.name, [multi_size_min, qwen2_vl_dim],
                      [multi_size_opt, qwen2_vl_dim],
                      [multi_size_max, qwen2_vl_dim])
    pos_ids.shape = [-1, 2]
    profile.set_shape(pos_ids.name, [multi_size_min, 2],
                      [multi_size_opt, 2],
                      [multi_size_max, 2])
    image_grid_thw.shape = [-1, 3]
    profile.set_shape(image_grid_thw.name, [min_batch_size, 3],
                      [opt_batch_size, 3],
                      [max_batch_size, 3])
    attenstion_mask_vit.shape = [1, -1, -1]
    profile.set_shape(attenstion_mask_vit.name,
                      [1, multi_size_min, multi_size_min],
                      [1, multi_size_opt, multi_size_opt],
                      [1, multi_size_max, multi_size_max])
    mrope_position_ids_padding.shape = [1, 3, 32768]
    profile.set_shape(mrope_position_ids_padding.name, [1, 3, 32768], [1, 3, 32768], [1, 3, 32768])
    mrope_position_deltas.shape = [1, 1]
    profile.set_shape(mrope_position_deltas.name, [1, 1], [1, 1], [1, 1])
    config.add_optimization_profile(profile)

    t0 = time.time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time.time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_path))
    else:
        logger.log(trt.Logger.INFO,
                   "Succeeded building %s in %d s" % (engine_path, t1 - t0))
        with open(engine_path, 'wb') as f:
            f.write(engine_string)

        # Clear onnx files since we no longer need them after a successful engine build
        if delete_onnx:
            shutil.rmtree(onnx_file)


def build_mrope_only_trt_engine(onnx_file,
                                engine_path,
                                dtype=torch.float16,
                                num_frames=None,
                                delete_onnx=False,
                                logger=trt.Logger(trt.Logger.VERBOSE)):
    logger.log(trt.Logger.INFO, f"Building TRT engine to {engine_path}")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config_args = {
        "precision": torch_dtype_to_str(dtype),
        "strongly_typed": False
    }
    if num_frames is not None:
        config_args["num_frames"] = num_frames

    config_wrapper = Builder().create_builder_config(**config_args)
    config = config_wrapper.trt_builder_config
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    mrope_position_ids_padding = network.get_input(0)
    mrope_position_deltas = network.get_input(1)

    mrope_position_ids_padding.shape = [1, 3, 32768]
    profile.set_shape(mrope_position_ids_padding.name, [1, 3, 32768], [1, 3, 32768], [1, 3, 32768])
    mrope_position_deltas.shape = [1, 1]
    profile.set_shape(mrope_position_deltas.name, [1, 1], [1, 1], [1, 1])
    config.add_optimization_profile(profile)

    t0 = time.time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time.time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_path))
    else:
        logger.log(trt.Logger.INFO,
                   "Succeeded building %s in %d s" % (engine_path, t1 - t0))
        with open(engine_path, 'wb') as f:
            f.write(engine_string)

        # Clear onnx files since we no longer need them after a successful engine build
        if delete_onnx:
            shutil.rmtree(onnx_file)


def load_trt_engine(engine_path):
    """load tensorrt engine and create execution context."""
    print('Loading tensorrt engine...')
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()  # create execution context
    inp_names = []
    out_names = []
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
            inp_names.append(name)
        else:
            out_names.append(name)

    return engine, context, inp_names, out_names


def trt_infer(inp_names, out_names, trt_ctx, stream, image, pos_ids,
              image_grid_thw, attention_mask_vit, mrope_position_ids_padding, mrope_position_deltas):
    """pure-trt infer."""
    """malloc tensorrt input and output cpu and gpu memory."""
    h_image = image.cpu()
    h_pos_ids = pos_ids.cpu()
    h_image_grid_thw = image_grid_thw.cpu()
    h_attention_mask_vit = attention_mask_vit.cpu()
    h_mrope_position_ids_padding = mrope_position_ids_padding.cpu()
    h_mrope_position_deltas = mrope_position_deltas.cpu()

    # Allocate device memory for inputs and outputs.
    d_image = cuda.mem_alloc(h_image.nbytes)  # float16
    d_pos_ids = cuda.mem_alloc(h_pos_ids.nbytes)  # int64
    d_image_grid_thw = cuda.mem_alloc(h_image_grid_thw.nbytes)  # int64
    d_attention_mask_vit = cuda.mem_alloc(h_attention_mask_vit.nbytes)  # bool
    d_mrope_position_ids_padding = cuda.mem_alloc(h_mrope_position_ids_padding.nbytes)  # int64
    d_mrope_position_deltas = cuda.mem_alloc(h_mrope_position_deltas.nbytes)  # int64

    # set true input shape
    print('h_image.shape: ', h_image.shape)
    print('h_image.value: ', h_image)
    trt_ctx.set_input_shape(inp_names[0], list(h_image.shape))
    print('h_pos_ids.shape: ', h_pos_ids.shape)
    print('h_pos_ids.value: ', h_pos_ids)
    trt_ctx.set_input_shape(inp_names[1], list(h_pos_ids.shape))
    print('h_image_grid_thw.shape: ', h_image_grid_thw.shape)
    print('h_image_grid_thw.value: ', h_image_grid_thw)
    trt_ctx.set_input_shape(inp_names[2], list(h_image_grid_thw.shape))
    print('h_attention_mask_vit.shape: ', h_attention_mask_vit.shape)
    print('h_attention_mask_vit.value: ', h_attention_mask_vit)

    trt_ctx.set_input_shape(inp_names[3], list(h_attention_mask_vit.shape))
    print('h_mrope_position_ids_padding.shape: ', h_mrope_position_ids_padding.shape)
    print('h_mrope_position_ids_padding.value: ', h_mrope_position_ids_padding)
    trt_ctx.set_input_shape(inp_names[4], list(h_mrope_position_ids_padding.shape))
    print('h_mrope_position_deltas.shape: ', h_mrope_position_deltas.shape)
    print('h_mrope_position_deltas.value: ', h_mrope_position_deltas)
    trt_ctx.set_input_shape(inp_names[5], list(h_mrope_position_deltas.shape))
    print('all_binding_shapes_specified: ', trt_ctx.all_binding_shapes_specified)

    # get true output shape.
    d_img_features_shape = trt_ctx.get_tensor_shape(out_names[0])
    print('d_img_features.shape: ', d_img_features_shape)
    d_img_features = cuda.mem_alloc(trt.volume(d_img_features_shape) * 2)  # float16
    d_concat_cos_sin_shape = trt_ctx.get_tensor_shape(out_names[1])
    print('d_concat_cos_sin.shape: ', d_concat_cos_sin_shape)
    d_concat_cos_sin = cuda.mem_alloc(trt.volume(d_concat_cos_sin_shape) * 4)  # float32
    d_mrope_position_deltas_out_shape = trt_ctx.get_tensor_shape(out_names[2])
    print('d_mrope_position_deltas_out.shape: ', d_mrope_position_deltas_out_shape)
    d_mrope_position_deltas_out = cuda.mem_alloc(trt.volume(d_mrope_position_deltas_out_shape) * 8)  # int64

    # copy input data from cpu to gpu
    cuda.memcpy_htod_async(d_image, h_image.numpy(), stream)
    cuda.memcpy_htod_async(d_pos_ids, h_pos_ids.numpy(), stream)
    cuda.memcpy_htod_async(d_image_grid_thw, h_image_grid_thw.numpy(), stream)
    cuda.memcpy_htod_async(d_attention_mask_vit, h_attention_mask_vit.numpy(), stream)
    cuda.memcpy_htod_async(d_mrope_position_ids_padding, h_mrope_position_ids_padding.numpy(), stream)
    cuda.memcpy_htod_async(d_mrope_position_deltas, h_mrope_position_deltas.numpy(), stream)

    # execute trt engine
    trt_ctx.set_tensor_address(inp_names[0], int(d_image))
    trt_ctx.set_tensor_address(inp_names[1], int(d_pos_ids))
    trt_ctx.set_tensor_address(inp_names[2], int(d_image_grid_thw))
    trt_ctx.set_tensor_address(inp_names[3], int(d_attention_mask_vit))
    trt_ctx.set_tensor_address(inp_names[4], int(d_mrope_position_ids_padding))
    trt_ctx.set_tensor_address(inp_names[5], int(d_mrope_position_deltas))
    trt_ctx.set_tensor_address(out_names[0], int(d_img_features))
    trt_ctx.set_tensor_address(out_names[1], int(d_concat_cos_sin))
    trt_ctx.set_tensor_address(out_names[2], int(d_mrope_position_deltas_out))
    trt_ctx.execute_async_v3(stream_handle=stream.handle)

    # copy output data from gpu to cpu
    h_img_features = cuda.pagelocked_empty(list(d_img_features_shape), dtype=np.float16)
    cuda.memcpy_dtoh_async(h_img_features, d_img_features, stream)
    h_concat_cos_sin = cuda.pagelocked_empty(list(d_concat_cos_sin_shape), dtype=np.float32)
    cuda.memcpy_dtoh_async(h_concat_cos_sin, d_concat_cos_sin, stream)
    h_mrope_position_deltas_out = cuda.pagelocked_empty(list(d_mrope_position_deltas_out_shape), dtype=np.int64)
    cuda.memcpy_dtoh_async(h_mrope_position_deltas_out, d_mrope_position_deltas_out, stream)
    # synchronize stream
    stream.synchronize()
    print('h_img_features.shape: ', h_img_features.shape)
    print('h_concat_cos_sin.shape: ', h_concat_cos_sin.shape)
    print('h_mrope_position_deltas_out.shape: ', h_mrope_position_deltas_out.shape)
    return h_img_features, h_concat_cos_sin, h_mrope_position_deltas_out


def compare_output(img_features_hf, concat_cos_sin_hf, img_features_trt, concat_cos_sin_trt):
    img_features_hf = img_features_hf.detach().cpu().numpy().astype(np.float16)
    img_features_trt = img_features_trt.astype(np.float16)
    for i in range(img_features_hf.shape[0]):
        diff = np.abs(img_features_hf[i] - img_features_trt[i])
        diff_sum = np.sum(diff)
        origin_sum = np.sum(np.abs(img_features_hf[i]))
        print("img_features {}, diff rate: {:.2f}%， diff sum: {}, origin_sum: {}".format(i, diff_sum / origin_sum * 100,
                                                                                         diff_sum, origin_sum))
    concat_cos_sin_hf = concat_cos_sin_hf.detach().numpy().astype(np.float32)
    concat_cos_sin_trt = concat_cos_sin_trt.astype(np.float32)
    diff = np.abs(concat_cos_sin_hf - concat_cos_sin_trt)
    diff_sum = np.sum(diff)
    origin_sum = np.sum(np.abs(concat_cos_sin_hf))
    print("concat_cos_sin diff rate: {:.2f}%， diff sum: {}, origin_sum: {}".format(diff_sum / origin_sum * 100,
                                                                                   diff_sum, origin_sum))


def parse_arguments():
    parser = argparse.ArgumentParser()
    # onnx/visual_encoder
    parser.add_argument('--onnxFile',
                        type=str,
                        default='visual_encoder/visual_encoder.onnx',
                        help='visiual encoder onnx file.')
    parser.add_argument('--mropeOnnxFile',
                        type=str,
                        default='visual_encoder/mrope.onnx',
                        help='mrope only onnx file.')
    parser.add_argument('--pretrainedModelPath',
                        type=str,
                        default='Qwen/Qwen2-VL-2B-Instruct',
                        help='')
    parser.add_argument('--trtFile',
                        type=str,
                        default='plan/visual_encoder/visual_encoder_fp16.plan',
                        help='visiual encoder trt file.')
    parser.add_argument('--mropeTrtFile',
                        type=str,
                        default='plan/visual_encoder/mrope_fp16.plan',
                        help='mrope only trt file.')
    parser.add_argument('--cvtOnnx',
                        action='store_true',
                        help='Run convert the hf to onnx format.')
    parser.add_argument('--cvtTrt',
                        action='store_true',
                        help='Run convert the onnx to TRT engine.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        help='The dtype of the model.',
                        choices=['bfloat16', 'float16'])
    parser.add_argument('--minBS', type=int, default=1)
    parser.add_argument('--optBS', type=int, default=1)
    parser.add_argument('--maxBS', type=int, default=4)
    parser.add_argument('--minHwDims', type=int, default=1, help=
    'Minimum multiply of h and w after patching for input images for qwen2_vl')
    parser.add_argument('--optHwDims', type=int, default=32 * 32, help=
    'Optimal multiply of h and w after patching for input images for qwen2_vl')
    parser.add_argument('--maxHwDims', type=int, default=128 * 128, help=
    'Maximum multiply of h and w after patching for input images for qwen2_vl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()

    min_pixels = 4 * 28 * 28
    max_pixels = 1024 * 1024 / 4
    hf_config = AutoConfig.from_pretrained(args.pretrainedModelPath)
    qwen2_vl_dim = hf_config.vision_config.in_chans * hf_config.vision_config.patch_size * hf_config.vision_config.patch_size * hf_config.vision_config.temporal_patch_size
    processor = AutoProcessor.from_pretrained(args.pretrainedModelPath, min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "file://./data/image1.jpg",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {
                    "type": "image",
                    "image": "file://./data/image2.jpg",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": "描述一下两张图片的不同。"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    image = inputs['pixel_values'].to(str_dtype_to_torch(args.dtype))
    image_grid_thw = inputs['image_grid_thw']

    rotary_pos_ids = compute_rotary_pos_emb(image_grid_thw, hf_config)

    cu_seqlens = torch.repeat_interleave(
        image_grid_thw[:, 1] * image_grid_thw[:, 2],
        image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    seq_length = image.shape[0]
    attention_mask_vit = torch.zeros([1, seq_length, seq_length],
                                     dtype=torch.bool)
    for i in range(1, len(cu_seqlens)):
        attention_mask_vit[..., cu_seqlens[i - 1]:cu_seqlens[i],
        cu_seqlens[i - 1]:cu_seqlens[i]] = True

    # generate mrope_params
    mrope_position_ids, mrope_position_deltas = get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )
    mrope_position_ids = mrope_position_ids.transpose(1, 0)
    mrope_position_ids_padding = torch.zeros([1, 3, 32768],
                                             dtype=torch.int32)
    mrope_position_ids_padding[:, :, :mrope_position_ids.shape[-1]] = mrope_position_ids

    if args.cvtOnnx:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.pretrainedModelPath,
            torch_dtype=str_dtype_to_torch(args.dtype),
            device_map="cpu")
        head_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads
        vision_encoder = VisionEncoderWrapper(model, hf_config, str_dtype_to_torch(args.dtype),
                                              VisionRotaryEmbedding(head_dim // 2)).cuda()

        dynamic_axes = {
            'image': {
                0: 'hw'
            },
            'rotary_pos_ids': {
                0: 'hw'
            },
            'image_grid_thw': {
                0: 'batch'
            },
            'attention_mask_vit': {
                1: 'hw',
                2: 'hw'
            },
        }
        export_onnx(vision_encoder,
                    (image.cuda(), rotary_pos_ids.cuda(), image_grid_thw.cuda(), attention_mask_vit.cuda(),
                     mrope_position_ids_padding, mrope_position_deltas.cuda()),
                    args.onnxFile,
                    input_names=['image', 'rotary_pos_ids', 'image_grid_thw', 'attention_mask_vit',
                                 'mrope_position_ids_padding', 'mrope_position_deltas'],
                    output_names=['img_features', 'concat_cos_sin', 'mrope_position_deltas_out'],
                    dynamic_axes=dynamic_axes)

        mrope_only = MropeOnlyWrapper(hf_config).cuda()
        export_onnx(mrope_only,
                    (mrope_position_ids_padding, mrope_position_deltas.cuda()),
                    args.mropeOnnxFile,
                    input_names=['mrope_position_ids_padding', 'mrope_position_deltas'],
                    output_names=['concat_cos_sin', 'mrope_position_deltas_out'])

        # HF infer
        print('image shape: {}, dtype: {}, val: {}'.format(image.shape, image.dtype, image))
        print('rotary_pos_ids shape: {}, dtype: {}, val: {}'.format(rotary_pos_ids.shape, rotary_pos_ids.dtype,
                                                                    rotary_pos_ids))
        print('max_grid_size shape: {}, dtype: {}, val: {}'.format(image_grid_thw.shape, image_grid_thw.dtype,
                                                                   image_grid_thw))
        print('attention_mask_vit shape: {}, dtype: {}, val: {}'.format(attention_mask_vit.shape,
                                                                        attention_mask_vit.dtype,
                                                                        attention_mask_vit))
        print('mrope_position_ids_padding shape: {}, dtype: {}, val: {}'.format(mrope_position_ids_padding.shape,
                                                                                mrope_position_ids_padding.dtype,
                                                                                mrope_position_ids_padding))
        print('mrope_position_deltas shape: {}, dtype: {}, val: {}'.format(mrope_position_deltas.shape,
                                                                           mrope_position_deltas.dtype,
                                                                           mrope_position_deltas))
        img_features_hf, concat_cos_sin_hf, mrope_position_deltas_hf = vision_encoder(image.cuda(),
                                                                                      rotary_pos_ids.cuda(),
                                                                                      image_grid_thw.cuda(),
                                                                                      attention_mask_vit.cuda(),
                                                                                      mrope_position_ids_padding,
                                                                                      mrope_position_deltas.cuda())
        print('HF img_features shape: {}, dtype: {}, val: {}'.format(img_features_hf.shape, img_features_hf.dtype,
                                                                     img_features_hf))
        print('HF concat_cos_sin shape: {}, dtype: {}, val: {}'.format(concat_cos_sin_hf.shape, concat_cos_sin_hf.dtype,
                                                                       concat_cos_sin_hf))
        print('HF mrope_position_deltas shape: {}, dtype: {}, val: {}'.format(mrope_position_deltas_hf.shape,
                                                                              mrope_position_deltas_hf.dtype,
                                                                              mrope_position_deltas_hf))
    if args.cvtTrt:
        build_trt_engine(onnx_file=args.onnxFile,
                         engine_path=args.trtFile,
                         dtype=str_dtype_to_torch(args.dtype),
                         qwen2_vl_dim=qwen2_vl_dim,
                         min_batch_size=args.minBS,
                         opt_batch_size=args.optBS,
                         max_batch_size=args.maxBS,
                         min_hw_dims=args.minHwDims,
                         opt_hw_dims=args.optHwDims,
                         max_hw_dims=args.maxHwDims)

        build_mrope_only_trt_engine(onnx_file=args.mropeOnnxFile,
                                    engine_path=args.mropeTrtFile,
                                    dtype=str_dtype_to_torch(args.dtype))

        if args.dtype == 'float16':  # numpy not support bfloat16
            # TRT infer
            import pycuda.autoinit

            trt_engine, trt_ctx, inp_names, out_names = load_trt_engine(args.trtFile)
            stream = cuda.Stream()
            img_features_trt, concat_cos_sin_trt, mrope_position_deltas_trt = trt_infer(inp_names, out_names, trt_ctx,
                                                                                        stream, image, rotary_pos_ids,
                                                                                        image_grid_thw,
                                                                                        attention_mask_vit,
                                                                                        mrope_position_ids_padding,
                                                                                        mrope_position_deltas)
            print(
                "TRT img_features shape: {}, dtype: {}, val: {}".format(img_features_trt.shape, img_features_trt.dtype,
                                                                        img_features_trt))
            print("TRT concat_cos_sin shape: {}, dtype: {}, val: {}".format(concat_cos_sin_trt.shape,
                                                                            concat_cos_sin_trt.dtype,
                                                                            concat_cos_sin_trt))
            print("TRT mrope_position_deltas shape: {}, dtype: {}, val: {}".format(mrope_position_deltas_trt.shape,
                                                                                   mrope_position_deltas_trt.dtype,
                                                                                   mrope_position_deltas_trt))

    # compare_output(img_features_hf, concat_cos_sin_hf, img_features_trt, concat_cos_sin_trt)
