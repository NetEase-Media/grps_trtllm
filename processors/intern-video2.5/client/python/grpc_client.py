# Grpc client demo. Complete interface description can be learned from docs/2_Interface.md
import os
import sys
import time
import ctypes
import mmap

import grpc
from grps_apis.grps_pb2 import GrpsMessage, GenericTensor, DataType
from grps_apis.grps_pb2_grpc import GrpsServiceStub

O_CREAT = 0o100
O_RDWR = 0o2
PROT_WRITE = 0x2
MAP_SHARED = 0x01
SHM_SIZE = 1024 * 1024 * 512

libc = ctypes.CDLL("libc.so.6")


def grpc_request(server, video_url):
    conn = grpc.insecure_channel(server,
                                 options=[('grpc.max_receive_message_length', 1024 * 1024 * 50)]  # 设置为 50MB
                                 )
    client = GrpsServiceStub(channel=conn)

    # predict with gmap.
    begin = time.time()
    request = GrpsMessage()
    request.gmap.s_s['video_url'] = video_url
    request.gmap.s_i32['max_frames'] = 128
    response = client.Predict(request)
    end = time.time()
    print(f'Response: {response}')
    print(f'Inference time: {end - begin:.2f}s')

    shm_path = response.gtensors.tensors[1].flat_string[0]
    print(f'Shared memory path: {shm_path}')
    fd = libc.shm_open(shm_path.encode('utf-8'), O_CREAT | O_RDWR, 0o666)
    if fd < 0:
        error_code = ctypes.get_errno()
        print(f"Error code: {error_code}")
        print(f"Error message: {os.strerror(error_code)}")
        raise Exception(f'Failed to create shared memory: {shm_path}')
    shm_mmap = mmap.mmap(fd, SHM_SIZE, flags=MAP_SHARED, prot=PROT_WRITE)
    # write flag 0 to first byte, stand for msg has been read.
    shm_mmap.seek(0)
    shm_mmap[0] = 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 grpc_client.py <server> <video_url>')
        sys.exit(1)

    grpc_request(sys.argv[1], sys.argv[2])
