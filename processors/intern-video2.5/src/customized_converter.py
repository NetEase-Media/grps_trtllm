# Customized converter of model, including pre-process and post-process.

import ctypes
import mmap
import os
import random
import shutil
import sys
import threading
import time

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import InterpolationMode

# Add src dir to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grps_framework.apis.grps_pb2 import DataType, GrpsMessage, GenericTensor
from grps_framework.context.context import GrpsContext
from grps_framework.converter.converter import Converter, converter_register
from grps_framework.logger.logger import clogger

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
O_CREAT = 0o100
O_RDWR = 0o2
PROT_WRITE = 0x2
MAP_SHARED = 0x01

libc = ctypes.CDLL("libc.so.6")


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                           T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                           T.Normalize(mean=MEAN, std=STD)])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
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


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def get_num_frames_by_duration(duration):
    local_num_frames = 4
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments

    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)

    return num_frames


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def download_video(video_url, video_path):
    response = requests.get(video_url)
    if response.status_code == 200:
        with open(video_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        raise Exception('Download video failed from url: {}'.format(video_url))


class YourConverter(Converter):
    """Your converter."""

    def __init__(self):
        super().__init__()
        self.video_cache_dir = '/tmp/intern-video2.5-processor/video'
        self.shm_pre = '/intern-video2.5-shm'
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

        # clean and create
        shutil.rmtree(self.video_cache_dir, ignore_errors=True)
        os.makedirs(self.video_cache_dir)

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
        video_url = inp.gmap.s_s.get('video_url', '')
        max_frames = inp.gmap.s_i32.get('max_frames', 128)
        if max_frames <= 0 or max_frames % 4 != 0:
            raise Exception('max_frames must be positive and multiple of 4, but got: {}'.format(max_frames))

        if video_url.startswith('http://') or video_url.startswith('https://'):
            # download video from url
            video_name = video_url.split('/')[-1] + time.strftime('-%Y%m%d%H%M%S-') + str(random.randint(1000, 9999))
            if not os.path.exists(self.video_cache_dir):
                os.makedirs(self.video_cache_dir)
            video_path = os.path.join(self.video_cache_dir, video_name)
            download_video(video_url, video_path)
            # save video path to context for delete video file later.
            context.put_user_data('video_path', video_path)
        elif video_url.startswith('file://'):
            video_path = video_url[7:]
            if not os.path.exists(video_path):
                raise Exception('Video file not exists: {}'.format(video_path))
        else:
            raise Exception('Invalid video url: {}'.format(video_url))

        # load video
        pixel_values, num_patches_list = load_video(video_path, num_segments=max_frames, max_num=1,
                                                    get_frame_by_duration=False)
        # save num_patches_list to context for return to client.
        context.put_user_data('num_patches_list', num_patches_list)
        return pixel_values

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
        tensor = inp.cpu()
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
            # check first byte to wait for client read, max wait 1s.
            retry = 0
            while self.shm_mmap[shm_idx][0] != 0 and retry < 300:
                time.sleep(0.01)
                retry += 1
            if retry >= 300:
                raise Exception('Wait client read shm timeout(3s), please check client svc and restart current '
                                'service, shm_path: {}'.format(self.shm_path[shm_idx]))

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
        num_patches_list = context.get_user_data('num_patches_list')
        gtensor = GenericTensor(name='num_patches_list')
        gtensor.shape.extend([len(num_patches_list)])
        gtensor.dtype = DataType.DT_INT32
        gtensor.flat_int32.extend(num_patches_list)
        out.gtensors.tensors.append(gtensor)
        gtensor = GenericTensor(name='shm_path')
        gtensor.shape.extend([1])
        gtensor.dtype = DataType.DT_STRING
        gtensor.flat_string.append(self.shm_path[shm_idx])
        out.gtensors.tensors.append(gtensor)
        gtensor = GenericTensor(name='tensor_shape')
        gtensor.shape.extend([len(inp.shape)])
        gtensor.dtype = DataType.DT_INT32
        gtensor.flat_int32.extend(inp.shape)
        out.gtensors.tensors.append(gtensor)

        # Delete video file if download from url.
        video_path = context.get_user_data('video_path')
        if video_path is not None:
            os.remove(video_path)
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
