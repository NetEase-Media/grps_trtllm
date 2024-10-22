# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

import tensorrt_llm
import torch
from tensorrt_llm._utils import (torch_dtype_to_str, pad_vocab_size)
from tensorrt_llm.layers import (ColumnLinear)
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraConfig, use_lora
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import PhiForCausalLM
from tensorrt_llm.models.modeling_utils import (DecoderModelForCausalLM, PretrainedConfig, QuantConfig)
from tensorrt_llm.models.phi3.convert import load_weights_from_hf_model
from tensorrt_llm.models.phi3.model import Phi3Model
from tensorrt_llm.quantization import QuantAlgo
from transformers import (AutoConfig, AutoModelForCausalLM)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
             'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()

    return args


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


def args_to_quant_config(args: argparse.Namespace) -> QuantConfig:
    '''return config dict with quantization info based on the command line args
    '''
    quant_config = QuantConfig()
    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            quant_config.quant_algo = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            quant_config.quant_algo = QuantAlgo.W4A16

    return quant_config


class Phi3Config(PretrainedConfig):

    def __init__(self,
                 *,
                 rotary_base: float = 10000.0,
                 rotary_scaling: Optional[dict] = None,
                 **kwargs):

        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in PhiConfig

        output['rotary_base'] = self.rotary_base
        output['rotary_scaling'] = self.rotary_scaling

        return output

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)

            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=True).llm_config

        num_key_value_heads = getattr(hf_config, "num_key_value_heads",
                                      hf_config.num_attention_heads)
        if dtype == 'auto':
            dtype = getattr(hf_config, 'torch_dtype', None)
            if dtype is None:
                dtype = 'float16'
            if isinstance(dtype, torch.dtype):
                dtype = torch_dtype_to_str(dtype)
            if dtype == 'float32':
                dtype = 'float16'
        if dtype == 'bfloat16' and torch.cuda.get_device_properties(
                0).major < 8:
            logger.warning(
                "Pre SM 80 GPUs do not support bfloat16, fallback to float16")
            dtype = 'float16'

        small_variant = hf_config.architectures[0] == "Phi3SmallForCausalLM"
        if small_variant:
            kwargs['gegelu_limit'] = getattr(hf_config, "gegelu_limit", None)
            kwargs['rotary_base'] = hf_config.rope_embedding_base
            kwargs['mup_attn_multiplier'] = getattr(hf_config,
                                                    "mup_attn_multiplier", None)
            kwargs['mup_embedding_multiplier'] = getattr(
                hf_config, "mup_embedding_multiplier", None)
            kwargs['mup_use_scaling'] = getattr(hf_config, "mup_use_scaling",
                                                None)
            kwargs['mup_width_multiplier'] = getattr(hf_config,
                                                     "mup_width_multiplier",
                                                     None)
            kwargs['blocksparse_block_size'] = getattr(
                hf_config, "blocksparse_block_size", None)
            kwargs['blocksparse_homo_head_pattern'] = getattr(
                hf_config, "blocksparse_homo_head_pattern", None)
            kwargs['blocksparse_num_local_blocks'] = getattr(
                hf_config, "blocksparse_num_local_blocks", None)
            kwargs['blocksparse_vertical_stride'] = getattr(
                hf_config, "blocksparse_vert_stride", None)
            kwargs['dense_attention_every_n_layers'] = getattr(
                hf_config, "dense_attention_every_n_layers", None)
        else:
            kwargs['rotary_base'] = hf_config.rope_theta
            kwargs['norm_epsilon'] = hf_config.rms_norm_eps
        kwargs['position_embedding_type'] = 'rope_gpt_neox'
        if hf_config.max_position_embeddings >= 128000:
            kwargs[
                'original_max_position_embeddings'] = hf_config.original_max_position_embeddings
            kwargs['position_embedding_type'] = "long_rope"
            kwargs['longrope_scaling_short_factors'] = hf_config.rope_scaling[
                "short_factor"]
            kwargs['longrope_scaling_long_factors'] = hf_config.rope_scaling[
                "long_factor"]
            if small_variant:
                kwargs['longrope_long_mscale'] = hf_config.rope_scaling[
                    "long_mscale"]
                kwargs['longrope_short_mscale'] = hf_config.rope_scaling[
                    "short_mscale"]

        return cls(architecture=hf_config.architectures[0],
                   dtype=dtype,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_attention_heads=hf_config.num_attention_heads,
                   hidden_size=hf_config.hidden_size,
                   intermediate_size=hf_config.intermediate_size,
                   num_key_value_heads=num_key_value_heads,
                   vocab_size=hf_config.vocab_size,
                   max_position_embeddings=hf_config.max_position_embeddings,
                   hidden_act="swiglu"
                   if hf_config.hidden_act == 'silu' else hf_config.hidden_act,
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)


class Phi3ForCausalLM(DecoderModelForCausalLM):
    config_class = Phi3Config

    def __init__(self, config: PretrainedConfig):
        transformer = Phi3Model(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size,
                                           config.mapping.tp_size)

        lm_head = ColumnLinear(config.hidden_size,
                               vocab_size_padded,
                               bias=False,
                               dtype=config.dtype,
                               tp_group=config.mapping.tp_group,
                               tp_size=config.mapping.tp_size,
                               gather_output=True)
        self.trtllm_modules_to_hf_modules = {
            "attn_qkv": ["qkv_proj", "query_key_value"],
            "attn_dense": ["o_proj", "dense"],
            "mlp_h_to_4h": ["gate_up_proj", "up_proj"],
            "mlp_4h_to_h": "down_proj",
        }
        super().__init__(config, transformer, lm_head)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir
        config = Phi3Config.from_hugging_face(hf_config_or_dir,
                                              dtype=dtype,
                                              mapping=mapping,
                                              quant_config=quant_config,
                                              **kwargs)

        if not use_preloading:
            hf_model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir, torch_dtype="auto", trust_remote_code=True).language_model

        assert isinstance(hf_model, transformers.PreTrainedModel)

        weights = load_weights_from_hf_model(hf_model, config)

        model = cls(config)
        model.load(weights)
        return model

    def use_lora(self, lora_config: LoraConfig):
        use_lora(self, lora_config, self.trtllm_modules_to_hf_modules)


if __name__ == '__main__':
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    assert args.pp_size == 1, "Pipeline parallelism is not supported."

    tik = time.time()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_config = AutoConfig.from_pretrained(args.model_dir,
                                              trust_remote_code=True).llm_config
    model_type = model_config.architectures[0]
    supported_models = [
        'PhiForCausalLM', 'Phi3ForCausalLM', 'Phi3VForCausalLM',
        'Phi3SmallForCausalLM'
    ]

    if model_type not in supported_models:
        assert False, "Invalid model type"

    phi_model = Phi3ForCausalLM if model_type.find(
        'Phi3') != -1 else PhiForCausalLM

    hf_model = None

    override_fields = {}
    # override_fields.update(args_to_build_options(args))
    quant_config = args_to_quant_config(args)


    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=args.tp_size * args.pp_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        phi = phi_model.from_hugging_face(
            args.model_dir if hf_model is None else hf_model,
            args.dtype,
            mapping=mapping,
            quant_config=quant_config,
            **override_fields,
        )
        phi.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del phi


    execute(args.workers, [convert_and_save_rank] * args.tp_size * args.pp_size,
            args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
