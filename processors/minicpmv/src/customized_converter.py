# Customized converter of model, including pre-process and post-process.
import base64
import ctypes
import io
import json
import mmap
import os
import random
import shutil
import sys
import threading
import time

import requests
from PIL import Image

# Add src dir to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grps_framework.apis.grps_pb2 import DataType, GrpsMessage, GenericTensor
from grps_framework.context.context import GrpsContext
from grps_framework.converter.converter import Converter, converter_register
from grps_framework.logger.logger import clogger

O_CREAT = 0o100
O_RDWR = 0o2
PROT_WRITE = 0x2
MAP_SHARED = 0x01

libc = ctypes.CDLL("libc.so.6")


def download_image(image_url):
    response = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        return response.content
    else:
        clogger.error('Failed to download image from url: {}, response: {}'.format(image_url, response))
        raise Exception('Download image failed from url: {}'.format(image_url))


class YourConverter(Converter):
    """Your converter."""

    def __init__(self):
        super().__init__()
        self.shm_pre = '/minicpmv-shm'
        self.shm_size = 512 * 1024 * 1024  # shared memory size for per shm used for images embeddings transfer.
        self.shm_cnt = 2  # shared memory count for images embeddings transfer.
        self.shm_mmap = []
        self.shm_path = []
        self.shm_lock = []
        self.cur_shm_idx = 0
        self.cur_shm_idx_lock = threading.Lock()

    def init(self, path=None, args=None):
        """
        Init converter.

        Args:
            path: Path.
            args: More args.

        Raises:
            Exception: If init failed, can raise exception and exception will be caught by server and show error message
            to user when start service.
        """
        super().init(path, args)

        # create shm.
        self.shm_size = args.get('shm_size', self.shm_size)
        self.shm_cnt = args.get('shm_cnt', self.shm_cnt)
        self.shm_pre = args.get('shm_pre', self.shm_pre)
        for i in range(self.shm_cnt):
            shm_path = f'{self.shm_pre}_{i}'
            fd = libc.shm_open(shm_path.encode('utf-8'), O_CREAT | O_RDWR, 0o666)
            if fd < 0:
                error_code = ctypes.get_errno()
                clogger.info(f"Error code: {error_code}")
                clogger.info(f"Error message: {os.strerror(error_code)}")
                raise Exception(f'Failed to create shared memory: {shm_path}')
            libc.ftruncate(fd, self.shm_size)
            shm_mmap = mmap.mmap(fd, self.shm_size, flags=MAP_SHARED, prot=PROT_WRITE)
            shm_mmap.seek(0)
            shm_mmap[0] = 0
            self.shm_mmap.append(shm_mmap)
            self.shm_path.append(shm_path)
            self.shm_lock.append(threading.Lock())

        clogger.info('your converter init, path: {}, args: {}'.format(path, args))

    def preprocess(self, inp: GrpsMessage, context: GrpsContext):
        """
        Preprocess.

        Args:
            inp: Input message from client or previous model(multi model sequential mode).
            context: Grps context of current request.

        Returns:
            Pre-processed data which is input of model inferer.

        Raises:
            Exception: If preprocess failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        # msgs = [{'role': 'user', 'content': [image_url, question]}]
        # clogger.info('input: {}'.format(inp))
        json_body = json.loads(inp.str_data)
        if len(json_body) == 0:
            raise Exception('The input message list is empty.')

        # download image and load.
        for msg in json_body:
            if msg['role'] == 'user':
                if isinstance(msg['content'], str):
                    pass
                elif isinstance(msg['content'], list):
                    # "content": [
                    #     {
                    #         "type": "text",
                    #         "text": "What'\''s in this image?"
                    #     },
                    #     {
                    #         "type": "image_url",
                    #         "image_url": {
                    #             "url": "https://**"
                    #         }
                    #     }
                    # ]
                    for i, content in enumerate(msg['content']):
                        if content.get('type') == 'text':
                            msg['content'][i] = content['text']
                        elif content.get('type') == 'image_url':
                            image_url = content.get('image_url', {}).get('url')
                            if image_url.startswith('http://') or image_url.startswith('https://'):
                                image = download_image(image_url)
                                msg['content'][i] = Image.open(io.BytesIO(image)).convert('RGB')
                            elif image_url.startswith('file://'):
                                image_path = image_url[7:]
                                if not os.path.exists(image_path):
                                    raise Exception('image file not exists: {}'.format(image_path))
                                msg['content'][i] = Image.open(image_path).convert('RGB')
                            elif image_url.startswith('data:image/') and image_url.find(';base64,') != -1:
                                base64_str = image_url.split(';base64,')[1]
                                image_data = base64.b64decode(base64_str)
                                msg['content'][i] = Image.open(io.BytesIO(image_data)).convert('RGB')
                            else:
                                raise Exception('Invalid image url: {}'.format(image_url))
                        else:
                            raise Exception('Invalid user content type: {}'.format(content.get('type')))
                else:
                    raise Exception('Invalid content type: {}'.format(type(msg['content'])))

        return json_body

    def postprocess(self, inp, context: GrpsContext) -> GrpsMessage:
        """
        Postprocess.

        Args:
            inp: Input to be post-processed, which is output of model inferer.
            context: Grps context of current request.

        Returns:
            Post-processed data with GrpsMessage format to client or next model(multi model sequential mode).

        Raises:
            Exception: If postprocess failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        (input_ids, tensor) = inp

        # No image input.
        if tensor is None:
            out = GrpsMessage()
            input_ids = input_ids[0]
            gtensor = GenericTensor(name='input_ids')
            gtensor.shape.extend([len(input_ids)])
            gtensor.dtype = DataType.DT_INT32
            gtensor.flat_int32.extend(input_ids)
            out.gtensors.tensors.append(gtensor)
            gtensor = GenericTensor(name='shm_path')
            gtensor.shape.extend([1])
            gtensor.dtype = DataType.DT_STRING
            gtensor.flat_string.append("")
            out.gtensors.tensors.append(gtensor)
            gtensor = GenericTensor(name='tensor_shape')
            gtensor.shape.extend([])
            gtensor.dtype = DataType.DT_INT32
            gtensor.flat_int32.extend([])
            out.gtensors.tensors.append(gtensor)
            return out

        # clogger.info(f'input_ids: {input_ids}')
        # clogger.info(f'tensor: {tensor}')

        tensor = tensor.cpu()
        tensor_size = tensor.numel() * tensor.element_size()
        if tensor_size > self.shm_size:
            raise Exception(
                f'tensor size {tensor_size} is larger than shm size {self.shm_size}, please increase shm size.')

        # Get shared memory index.
        shm_idx = None
        with self.cur_shm_idx_lock:
            shm_idx = self.cur_shm_idx
            self.cur_shm_idx = (self.cur_shm_idx + 1) % self.shm_cnt

        with self.shm_lock[shm_idx]:
            # check first byte to wait for client read, max wait 3s.
            retry = 0
            while self.shm_mmap[shm_idx][0] != 0 and retry < 300:
                time.sleep(0.01)
                retry += 1
            if retry >= 300:
                clogger.warning(
                    f'shm {self.shm_path[shm_idx]} wait for client read timeout(3s), will overwrite.')

            # set first byte to 1 to stand for new data and not read by client.
            self.shm_mmap[shm_idx].seek(0)
            self.shm_mmap[shm_idx][0] = 1
            # write tensor data to shm.
            # clogger.info(tensor[0, 0, 0:10])
            # clogger.info(tensor[tensor.shape[0] - 1, tensor.shape[1] - 1, -10:])
            tensor_bytes = bytearray((ctypes.c_ubyte * tensor_size).from_address(tensor.data_ptr()))
            self.shm_mmap[shm_idx][1:tensor_size + 1] = tensor_bytes

        out = GrpsMessage()
        # gtensor = GenericTensor(name='img_embeddings')
        # gtensor.shape.extend(inp.shape)
        # gtensor.dtype = DataType.DT_FLOAT32
        # gtensor.flat_float32.extend(inp.cpu().flatten().tolist())
        # out.gtensors.tensors.append(gtensor)
        input_ids = input_ids[0]
        gtensor = GenericTensor(name='input_ids')
        gtensor.shape.extend([len(input_ids)])
        gtensor.dtype = DataType.DT_INT32
        gtensor.flat_int32.extend(input_ids)
        out.gtensors.tensors.append(gtensor)
        gtensor = GenericTensor(name='shm_path')
        gtensor.shape.extend([1])
        gtensor.dtype = DataType.DT_STRING
        gtensor.flat_string.append(self.shm_path[shm_idx])
        out.gtensors.tensors.append(gtensor)
        gtensor = GenericTensor(name='tensor_shape')
        gtensor.shape.extend([len(tensor.shape)])
        gtensor.dtype = DataType.DT_INT32
        gtensor.flat_int32.extend(tensor.shape)
        out.gtensors.tensors.append(gtensor)

        return out

    def batch_preprocess(self, inps: list, contexts: list):
        """
        Batch preprocess.

        Args:
            inps: Input messages from client or previous model(multi model sequential mode).
            contexts: Grps contexts of current requests.

        Returns:
            Pre-processed data which is input of model inferer.

        Raises:
            Exception: If preprocess failed, can raise exception and exception will be caught by server and return error
            message to client.
        """
        # You can preprocess every request and convert to tensor. Merge tensors of each request to batch tensor.
        # Add your codes here.
        pass

    def batch_postprocess(self, inp, contexts: list) -> list:
        """
        Batch postprocess.

        Args:
            inp: Input to be post-processed, which is output of model inferer.
            contexts: Grps contexts of current requests.

        Returns:
            Post-processed data with GrpsMessage format to client or next model(multi model sequential mode).

        Raises:
            Exception: If postprocess failed, can raise exception and exception will be caught by server and return
            error message to client.
        """
        # You can postprocess batch tensor and convert to response. Split batch tensor to tensors of each request.
        # Add your codes here.
        pass


converter_register.register('your_converter', YourConverter())
