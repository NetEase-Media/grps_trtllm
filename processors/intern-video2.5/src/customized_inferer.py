# Customized deep learning model inferer. Including model load and model infer.
import gc
import os
import sys

from transformers import AutoModelForCausalLM

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
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self._path,
            device_map='cpu',
            trust_remote_code=True,
            use_flash_attn=False
        ).eval().to(self.dtype)
        # free llm memory, only use vit.
        del self.hf_model.language_model
        gc.collect()
        self.hf_model.vision_model.to(self._device)
        self.hf_model.mlp1.to(self._device)
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
        with torch.no_grad():
            return self.hf_model.extract_feature(inp.to(self.dtype).to(self._device))

    def batch_infer(self, inp, contexts: list):
        pass


# Register
inferer_register.register('your_inferer', YourInferer())
