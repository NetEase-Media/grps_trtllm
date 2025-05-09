import base64
import os
import shutil
import socket
import sys
import uuid
from io import BytesIO

import gradio as gr
import numpy as np
import openai
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer

VIDEO_MAX_FRAMES = 8

SERVER_PORT = 7860

if len(sys.argv) < 3:
    print('python3 llm_app.py <app_type> <llm_server>')
    exit(1)

app_type = sys.argv[1]
llm_server = sys.argv[2]

if app_type == 'olm-ocr':
    from olmocr.data.renderpdf import render_pdf_to_base64png
    from olmocr.prompts import build_finetuning_prompt
    from olmocr.prompts.anchor import get_anchor_text


def get_ip_socket():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    img_urls = []
    # generate random uuid for current video dir.
    img_url_root = '/tmp/gradio/video/' + str(uuid.uuid4()) + '/'
    os.makedirs(img_url_root, exist_ok=True)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        img_url = img_url_root + 'frame' + str(frame_index) + '.jpg'
        img.save(img_url)
        img_urls.append(img_url)
    return img_urls, img_url_root


def llm_fn(message, history, max_tokens):
    # print('message:', message)
    # print('history:', history)

    # History messages.
    messages = []
    for his in history:
        messages.append({
            "role": his['role'],
            "content": his['content']
        })

    # New message.
    new_message = {
        "role": "user",
        "content": message
    }
    messages.append(new_message)
    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True,
        max_tokens=max_tokens
    )
    # print('res: ', res)

    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content
                yield content
    except openai.APIError as e:
        print('error:', e)
        yield 'error: ' + e.message
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


def deepseek_llm_fn(message, history, max_tokens):
    # print('message:', message)
    # print('history:', history)

    # History messages.
    messages = []
    for his in history:
        messages.append({
            "role": his['role'],
            "content": his['content'][8:].replace('\n```\n</think>', '</think>') if his['role'] == 'assistant' else his[
                'content']
        })

    # New message.
    new_message = {
        "role": "user",
        "content": message
    }
    messages.append(new_message)
    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True,
        max_tokens=max_tokens,

    )
    # print('res: ', res)
    content = '```text\n'
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content.replace('</think>', '\n```\n</think>')
                yield content
    except openai.APIError as e:
        print('error:', e)
        yield 'error: ' + e.message
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


def qwen3_llm_fn(message, history, max_tokens):
    # print('message:', message)
    # print('history:', history)

    # History messages.
    messages = []
    for his in history:
        messages.append({
            "role": his['role'],
            "content": his['content'][8:]
            .replace('<blockquote id="think" style="font-size:0.8em;">', '<think>')
            .replace('</blockquote id="think">', '</think>') if his['role'] == 'assistant' else his['content']
        })

    # New message.
    new_message = {
        "role": "user",
        "content": message
    }
    messages.append(new_message)
    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
    )
    # print('res: ', res)
    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += (msg.choices[0].delta.content
                            .replace('<think>',
                                     '<blockquote id="think" style="font-size:0.8em;">')
                            .replace('</think>', '</blockquote id="think">'))
                yield content
    except openai.APIError as e:
        print('error:', e)
        yield 'error: ' + e.message
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


def internvl2_llm_fn(message, history):
    # print('message:', message)
    # print('history:', history)

    img_dir = None

    # History messages.
    messages = []
    pre_messages = []
    for his in history:
        if his['role'] == 'user':
            pre_messages.append(his['content'])
        elif his['role'] == 'assistant':
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": pre_messages[-1]  # last is the text content
                    }
                ]
            }
            if len(pre_messages) > 1 and (pre_messages[0][0].endswith('.mp4') or pre_messages[0][0].endswith('.mov')
                                          or pre_messages[0][0].endswith('.avi')):  # video
                if len(pre_messages) > 2:
                    yield 'error: 视频文件个数不能超过1个。'
                    return
                try:
                    frames_urls, img_dir = load_video(pre_messages[0][0], num_segments=VIDEO_MAX_FRAMES)
                except Exception as e:
                    yield 'error: Load video failed, ' + str(e)
                    return
                for frame_url in frames_urls:
                    msg['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": 'file://' + frame_url
                        }
                    })

                # insert <image> in text start
                pre_text = ''
                for i in range(len(frames_urls)):
                    pre_text += 'Frame' + str(i + 1) + ': <image>\n'
                msg['content'][0]['text'] = pre_text + pre_messages[-1]
            else:  # images
                if len(pre_messages) > 1:
                    for pre_message in pre_messages[:-1]:  # image content
                        msg['content'].append({
                            "type": "image_url",
                            "image_url": {
                                "url": 'file://' + pre_message[0]
                            }
                        })
                image_flag_count = pre_messages[-1].count('<image>')
                if image_flag_count == 0:
                    # insert <image> in text start
                    if len(pre_messages[:-1]) == 1:
                        msg['content'][0]['text'] = '<image>\n' + pre_messages[-1]
                    elif len(pre_messages[:-1]) > 1:
                        pre_text = ''
                        for i in range(len(pre_messages[:-1])):
                            pre_text += 'Image-' + str(i + 1) + ': <image>\n'
                        msg['content'][0]['text'] = pre_text + pre_messages[-1]
                if 0 < image_flag_count != len(pre_messages[:-1]):
                    yield 'error: `<image>`与实际图片数量不一致。'
                    return

            messages.append(msg)
            messages.append({
                "role": "assistant",
                "content": his['content']
            })
            pre_messages = []

    # New message.
    new_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": message['text']
            }
        ]
    }
    if 'files' in message and len(message['files']) > 0 and (message['files'][0]['path'].endswith('.mp4')
                                                             or message['files'][0]['path'].endswith('.mov') or
                                                             message['files'][0]['path'].endswith('.avi')
    ):  # video
        if len(message['files']) > 1:
            yield 'error: 视频文件个数不能超过1个。'
            return
        try:
            frames_urls, img_dir = load_video(message['files'][0]['path'], num_segments=VIDEO_MAX_FRAMES)
        except Exception as e:
            yield 'error: Load video failed, ' + str(e)
            return
        for frame_url in frames_urls:
            new_message['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": 'file://' + frame_url
                }
            })

        messages.append(new_message)

        # insert <image> in text start
        pre_text = ''
        for i in range(len(frames_urls)):
            pre_text += 'Frame' + str(i + 1) + ': <image>\n'
        new_message['content'][0]['text'] = pre_text + message['text']
    else:  # images
        if 'files' in message:
            for file in message['files']:
                new_message['content'].append({
                    "type": "image_url",
                    "image_url": {
                        "url": 'file://' + file['path']
                    }
                })
        messages.append(new_message)

        image_flag_count = message['text'].count('<image>')
        if image_flag_count == 0:
            # insert <image> in text start
            if len(message['files']) == 1:
                new_message['content'][0]['text'] = '<image>\n' + message['text']
            elif len(message['files']) > 1:
                pre_text = ''
                for i in range(len(message['files'])):
                    pre_text += 'Image-' + str(i + 1) + ': <image>\n'
                new_message['content'][0]['text'] = pre_text + message['text']
        if 0 < image_flag_count != len(message['files']):
            yield 'error: `<image>`与实际图片数量不一致。'
            return

    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True
    )
    # print('res: ', res)

    if img_dir is not None:
        shutil.rmtree(img_dir)

    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content
                yield content
        # print('response:', content)
    except openai.APIError as e:
        print('error:', e)
        if '[TrtInfererException] Dims not match' in e.message:
            yield 'error: 图片尺寸过大或超过图片个数限制。'
        else:
            yield 'error: ' + e.message.replace('<image>', '`<image>`')
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


def intern_video_2_5_llm_fn(message, history):
    # print('message:', message)
    # print('history:', history)

    img_dir = None

    # History messages.
    messages = []
    pre_messages = []
    for his in history:
        if his['role'] == 'user':
            pre_messages.append(his['content'])
        elif his['role'] == 'assistant':
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": pre_messages[-1]  # last is the text content
                    }
                ]
            }
            if len(pre_messages) > 1 and (pre_messages[0][0].endswith('.mp4') or pre_messages[0][0].endswith('.mov')
                                          or pre_messages[0][0].endswith('.avi')):  # video
                if len(pre_messages) > 2:
                    yield 'error: 视频文件个数不能超过1个。'
                    return
                msg['content'].append({
                    "type": "video_url",
                    "video_url": {
                        "url": 'file://' + pre_messages[0][0]
                    }
                })

            messages.append(msg)
            messages.append({
                "role": "assistant",
                "content": his['content']
            })
            pre_messages = []

    # New message.
    new_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": message['text']
            }
        ]
    }
    if 'files' in message and len(message['files']) > 0 and (message['files'][0]['path'].endswith('.mp4')
                                                             or message['files'][0]['path'].endswith('.mov') or
                                                             message['files'][0]['path'].endswith('.avi')
    ):  # video
        if len(message['files']) > 1:
            yield 'error: 视频文件个数不能超过1个。'
            return
        new_message['content'].append({
            "type": "video_url",
            "video_url": {
                "url": 'file://' + message['files'][0]['path']
            }
        })

    messages.append(new_message)
    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True
    )
    # print('res: ', res)

    if img_dir is not None:
        shutil.rmtree(img_dir)

    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content
                yield content
        # print('response:', content)
    except openai.APIError as e:
        print('error:', e)
        yield 'error: ' + e.message.replace('<image>', '`<image>`')
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


qwenvl_tokenizer = None


def qwenvl_llm_fn(message, history):
    # print('message:', message)
    # print('history:', history)

    img_dir = None

    last_img_url = None
    # History messages.
    messages = []
    pre_messages = []
    for his in history:
        if his['role'] == 'user':
            pre_messages.append(his['content'])
        elif his['role'] == 'assistant':
            if len(pre_messages) == 0:
                continue
            msg = {
                "role": "user",
                "content": [
                ]
            }
            if len(pre_messages) > 1:
                for pre_message in pre_messages[:-1]:  # image content
                    msg['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": 'file://' + pre_message[0]
                        }
                    })
                    last_img_url = pre_message[0]
            msg['content'].append({
                "type": "text",
                "text": pre_messages[-1]  # last is the text content
            })
            messages.append(msg)
            messages.append({
                "role": "assistant",
                "content": his['content']
            })
            pre_messages = []

    # New message.
    new_message = {
        "role": "user",
        "content": [
        ]
    }
    if 'files' in message:
        for file in message['files']:
            new_message['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": 'file://' + file['path']
                }
            })
            last_img_url = file['path']
    new_message['content'].append({
        "type": "text",
        "text": message['text']
    })
    messages.append(new_message)

    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True
    )
    # print('res: ', res)

    if img_dir is not None:
        shutil.rmtree(img_dir)

    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content
                yield content
        # print('response:', content)
        if '<ref>' in content and '<box>' in content:
            # build history.
            tmp = [('Picture 1: <img>' + last_img_url + '</img>\n' + message['text'],
                    content)]
            # print('history:', tmp)
            image = qwenvl_tokenizer.draw_bbox_on_latest_picture(content, tmp)
            if image:
                img_url_root = '/tmp/gradio/box/' + str(uuid.uuid4()) + '/'
                os.makedirs(img_url_root, exist_ok=True)
                img_url = img_url_root + 'box.jpg'
                image.save(img_url)
                yield content + f'\n![image](http://{get_ip_socket()}:{SERVER_PORT}/file=' + img_url + ')'

    except openai.APIError as e:
        print('error:', e)
        if '[TrtInfererException] Dims not match' in e.message:
            yield 'error: 图片尺寸过大或超过图片个数限制。'
        else:
            yield 'error: ' + e.message
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


def qwen2vl_llm_fn(message, history):
    # print('message:', message)
    # print('history:', history)

    img_dir = None

    last_img_url = None
    # History messages.
    messages = []
    pre_messages = []
    for his in history:
        if his['role'] == 'user':
            pre_messages.append(his['content'])
        elif his['role'] == 'assistant':
            if len(pre_messages) == 0:
                continue
            msg = {
                "role": "user",
                "content": [
                ]
            }
            if len(pre_messages) > 1:
                for pre_message in pre_messages[:-1]:  # image content
                    msg['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": 'file://' + pre_message[0]
                        }
                    })
                    last_img_url = pre_message[0]
            msg['content'].append({
                "type": "text",
                "text": pre_messages[-1]  # last is the text content
            })
            messages.append(msg)
            messages.append({
                "role": "assistant",
                "content": his['content']
            })
            pre_messages = []

    # New message.
    new_message = {
        "role": "user",
        "content": [
        ]
    }
    if 'files' in message:
        for file in message['files']:
            new_message['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": 'file://' + file['path']
                }
            })
            last_img_url = file['path']
    new_message['content'].append({
        "type": "text",
        "text": message['text']
    })
    messages.append(new_message)

    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True
    )
    # print('res: ', res)

    if img_dir is not None:
        shutil.rmtree(img_dir)

    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content
                yield content
        # print('response:', content)

    except openai.APIError as e:
        print('error:', e)
        if '[TrtInfererException] Dims not match' in e.message:
            yield 'error: 图片尺寸过大或超过图片个数限制。'
        else:
            yield 'error: ' + e.message
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


def janus_pro_llm_fn(message, history):
    # print('message:', message)
    # print('history:', history)

    img_dir = None

    # History messages.
    messages = []
    pre_messages = []
    for his in history:
        if his['role'] == 'user':
            pre_messages.append(his['content'])
        elif his['role'] == 'assistant':
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": pre_messages[-1]  # last is the text content
                    }
                ]
            }
            if len(pre_messages) > 1:
                for pre_message in pre_messages[:-1]:  # image content
                    msg['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": 'file://' + pre_message[0]
                        }
                    })
            image_flag_count = pre_messages[-1].count('<image_placeholder>')
            if image_flag_count == 0:
                # insert <image_placeholder> in text start
                if len(pre_messages[:-1]) == 1:
                    msg['content'][0]['text'] = '<image_placeholder>\n' + pre_messages[-1]
                elif len(pre_messages[:-1]) > 1:
                    pre_text = ''
                    for i in range(len(pre_messages[:-1])):
                        pre_text += 'Image-' + str(i + 1) + ': <image_placeholder>\n'
                    msg['content'][0]['text'] = pre_text + pre_messages[-1]
            if 0 < image_flag_count != len(pre_messages[:-1]):
                yield 'error: `<image_placeholder>`与实际图片数量不一致。'
                return

            messages.append(msg)
            messages.append({
                "role": "assistant",
                "content": his['content']
            })
            pre_messages = []

    # New message.
    new_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": message['text']
            }
        ]
    }
    if 'files' in message:
        for file in message['files']:
            new_message['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": 'file://' + file['path']
                }
            })
    messages.append(new_message)

    image_flag_count = message['text'].count('<image_placeholder>')
    if image_flag_count == 0:
        # insert <image_placeholder> in text start
        if len(message['files']) == 1:
            new_message['content'][0]['text'] = '<image_placeholder>\n' + message['text']
        elif len(message['files']) > 1:
            pre_text = ''
            for i in range(len(message['files'])):
                pre_text += 'Image-' + str(i + 1) + ': <image_placeholder>\n'
            new_message['content'][0]['text'] = pre_text + message['text']
    if 0 < image_flag_count != len(message['files']):
        yield 'error: `<image_placeholder>`与实际图片数量不一致。'
        return

    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True
    )
    # print('res: ', res)

    if img_dir is not None:
        shutil.rmtree(img_dir)

    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content
                yield content
        # print('response:', content)
    except openai.APIError as e:
        print('error:', e)
        if '[TrtInfererException] Dims not match' in e.message:
            yield 'error: 图片尺寸过大或超过图片个数限制。'
        else:
            yield 'error: ' + e.message.replace('<image_placeholder>', '`<image_placeholder>`')
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


def olm_ocr_fn(message, history):
    # print('message:', message)
    # print('history:', history)

    if 'files' not in message or len(message['files']) != 1 or not message['files'][0]['path'].endswith('.pdf'):
        yield 'error: 一次仅支持单个pdf文件。'
        return

    pdf_path = message['files'][0]['path']
    num_pages = 0
    try:
        cmd = f'pdfinfo \'{pdf_path}\' | grep Pages | awk \'{{print $2}}\''
        num_pages = os.popen(cmd).read().strip()
        num_pages = int(num_pages)
    except Exception as e:
        yield 'error: ' + str(e)
        return

    response = ''
    for i in range(1, num_pages + 1):
        response += '\n# Page ' + str(i) + '\n'
        image_base64 = render_pdf_to_base64png(pdf_path, i, target_longest_image_dim=1024)
        # print('len(image_base64)', len(image_base64))
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        image.save(f'page_{i}.jpg')
        # Build the prompt, using document metadata
        anchor_text = get_anchor_text(pdf_path, i, pdf_engine="pdfreport", target_length=4000)
        prompt = build_finetuning_prompt(anchor_text)
        print(f'page{i}_prompt: {prompt}')

        # Request to openai llm server.
        new_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f'data:image/png;base64,{image_base64}'
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        client = openai.Client(
            api_key="cannot be empty",
            base_url=f"http://{llm_server}/v1"
        )
        res = client.chat.completions.create(
            model="",
            messages=[new_message],
            stream=True
        )
        # print('res: ', res)
        try:
            tmp = ''
            pre_len = 0
            for msg in res:
                # print('msg:', msg)
                if msg.choices[0].delta.content is not None:
                    tmp += msg.choices[0].delta.content
                    if tmp.find('"natural_text":"') == -1:
                        continue
                    if tmp.endswith('\\'):
                        continue
                text = tmp.split('"natural_text":"')[1].split('"}')[0].replace('\\n', '\n')
                response += text[pre_len:]
                pre_len = len(text)
                yield response

        except openai.APIError as e:
            print('error:', e)
            yield 'error: ' + e.message
        except Exception as e:
            print('error:', e)
            yield 'error: ' + str(e)


def minicpmv_llm_fn(message, history):
    # print('message:', message)
    # print('history:', history)

    img_dir = None

    # History messages.
    messages = []
    pre_messages = []
    for his in history:
        if his['role'] == 'user':
            pre_messages.append(his['content'])
        elif his['role'] == 'assistant':
            msg = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": pre_messages[-1]  # last is the text content
                    }
                ]
            }
            if len(pre_messages) > 1:
                for pre_message in pre_messages[:-1]:  # image content
                    msg['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": 'file://' + pre_message[0]
                        }
                    })

            messages.append(msg)
            messages.append({
                "role": "assistant",
                "content": his['content']
            })
            pre_messages = []

    # New message.
    new_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": message['text']
            }
        ]
    }
    if 'files' in message:
        for file in message['files']:
            new_message['content'].append({
                "type": "image_url",
                "image_url": {
                    "url": 'file://' + file['path']
                }
            })
    messages.append(new_message)

    # print('messages:', messages)

    # Request to openai llm server.
    client = openai.Client(
        api_key="cannot be empty",
        base_url=f"http://{llm_server}/v1"
    )

    res = client.chat.completions.create(
        model="",
        messages=messages,
        stream=True
    )
    # print('res: ', res)

    if img_dir is not None:
        shutil.rmtree(img_dir)

    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content
                yield content
        # print('response:', content)
    except openai.APIError as e:
        print('error:', e)
        yield 'error: ' + e.message
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


if app_type == 'llm':
    demo = gr.ChatInterface(
        concurrency_limit=32, fn=llm_fn, type="messages", title="grps-trtllm",
        additional_inputs=[
            gr.Slider(1, 32768, value=2048, step=1, label="Max Tokens"),
        ], examples=[
            ["你好，你是谁？", 2048],
            ["提供一段快速排序的c++代码：", 2048],
        ], multimodal=False)
elif app_type == 'internvl2':
    demo = gr.ChatInterface(concurrency_limit=32, fn=internvl2_llm_fn, type="messages", examples=[
        {"text": "你好，你是谁？"},
        {"text": "描述一下这张图片：",
         "files": ['https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg']},
        {"text": "描述一下这两张图片：",
         "files": [
             'https://p6.itc.cn/q_70/images03/20230821/69b103277521450e89090a24df1327d7.jpeg',
             'https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg']},
        {"text": "描述一下这个视频：",
         "files": [os.path.dirname(os.path.realpath(__file__)) + '/data/red-panda.mp4']},
    ],
                            title="internvl2-grps-trtllm",
                            multimodal=True)
elif app_type == 'intern-video2.5':
    demo = gr.ChatInterface(concurrency_limit=32, fn=intern_video_2_5_llm_fn, type="messages", examples=[
        {"text": "描述一下这个视频：",
         "files": [os.path.dirname(os.path.realpath(__file__)) + '/data/red-panda.mp4']},
    ],
                            title="intern-video2.5-grps-trtllm",
                            multimodal=True)
elif app_type == 'internvl3':
    demo = gr.ChatInterface(concurrency_limit=32, fn=internvl2_llm_fn, type="messages", examples=[
        {"text": "你好，你是谁？"},
        {"text": "解一下这道题：",
         "files": [os.path.dirname(os.path.realpath(__file__)) + '/data/leetcode_205.png']},
        {"text": "描述一下这两张图片：",
         "files": [
             'https://p6.itc.cn/q_70/images03/20230821/69b103277521450e89090a24df1327d7.jpeg',
             'https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg']},
        {"text": "描述一下这个视频：",
         "files": [os.path.dirname(os.path.realpath(__file__)) + '/data/red-panda.mp4']},
    ],
                            title="internvl3-grps-trtllm",
                            multimodal=True)
elif app_type == 'qwenvl':
    if not os.path.exists("/tmp/Qwen-VL-Chat"):
        print("Please download the qwenvl to /tmp/Qwen-VL-Chat first.")
        exit(1)
    qwenvl_tokenizer = AutoTokenizer.from_pretrained("/tmp/Qwen-VL-Chat", trust_remote_code=True)
    demo = gr.ChatInterface(concurrency_limit=32, fn=qwenvl_llm_fn, type="messages", examples=[
        {"text": "你好，你是谁？"},
        {"text": "描述一下两张图片的不同。",
         "files": [
             'https://p6.itc.cn/q_70/images03/20230821/69b103277521450e89090a24df1327d7.jpeg',
             'https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg']},
        {"text": "输出\"女生\"的检测框。",
         "files": ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg']},
    ],
                            title="qwenvl-grps-trtllm",
                            multimodal=True)
elif app_type == 'qwen2vl':
    demo = gr.ChatInterface(concurrency_limit=32, fn=qwen2vl_llm_fn, type="messages", examples=[
        {"text": "你好，你是谁？"},
        {"text": "描述一下这张图片：",
         "files": ['https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg']},
        {"text": "简述一下两张图片的不同。",
         "files": [
             'https://p6.itc.cn/q_70/images03/20230821/69b103277521450e89090a24df1327d7.jpeg',
             'https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg']},
    ],
                            title="qwen2vl-grps-trtllm",
                            multimodal=True)
elif app_type == 'deepseek-r1':
    demo = gr.ChatInterface(
        concurrency_limit=32, fn=deepseek_llm_fn, type="messages", title="deepseek-r1-grps-trtllm",
        multimodal=False, additional_inputs=[
            gr.Slider(1, 32768, value=2048, step=1, label="Max Tokens"),
        ],
        examples=[
            ["你好，你是谁？", 2048],
            ["解一下这道题：\n(x + 3) = (8 - x)\nx = ?", 2048]
        ])
elif app_type == 'qwq':
    demo = gr.ChatInterface(
        concurrency_limit=32, fn=deepseek_llm_fn, type="messages", title="qwq-grps-trtllm",
        multimodal=False, additional_inputs=[
            gr.Slider(1, 32768, value=2048, step=1, label="Max Tokens"),
        ],
        examples=[
            ["你好，你是谁？", 2048],
            ["解一下这道题：\n(x + 3) = (8 - x)\nx = ?", 2048]
        ])
elif app_type == 'qwen3':
    demo = gr.ChatInterface(
        concurrency_limit=32, fn=qwen3_llm_fn, type="messages", title="qwen3-grps-trtllm", fill_height=False,
        multimodal=False, additional_inputs=[
            gr.Slider(1, 32768, value=2048, step=1, label="Max Tokens"),
        ],
        examples=[
            ["你好，你是谁？", 2048],
            ["解一下这道题：\n(x + 3) = (8 - x)\nx = ?", 2048]
        ])
elif app_type == 'janus-pro':
    demo = gr.ChatInterface(concurrency_limit=32, fn=janus_pro_llm_fn, type="messages", examples=[
        {"text": "你好，你是谁？"},
        {"text": "这是什么？",
         "files": ['https://raw.githubusercontent.com/deepseek-ai/Janus/refs/heads/main/images/logo.png']},
        {"text": "描述一下这两张图片：",
         "files": [
             'https://p6.itc.cn/q_70/images03/20230821/69b103277521450e89090a24df1327d7.jpeg',
             'https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg']},
    ],
                            title="janus-pro-grps-trtllm",
                            multimodal=True)
elif app_type == 'olm-ocr':
    demo = gr.ChatInterface(concurrency_limit=32, fn=olm_ocr_fn, type="messages", examples=[
        {"text": "",
         "files": ['https://molmo.allenai.org/paper.pdf']},
    ],
                            title="olm-ocr-grps-trtllm",
                            multimodal=True)
elif app_type == 'minicpmv':
    demo = gr.ChatInterface(concurrency_limit=32, fn=minicpmv_llm_fn, type="messages", examples=[
        {"text": "你好，你是谁？"},
        {"text": "描述一下这张图片：",
         "files": ['https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg']},
        {"text": "描述一下这两张图片：",
         "files": [
             'https://p6.itc.cn/q_70/images03/20230821/69b103277521450e89090a24df1327d7.jpeg',
             'https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg']},
    ],
                            title="minicpmv-grps-trtllm",
                            multimodal=True)
else:
    print('`app_type` only support `llm`(text llm) or `internvl2`(multi-modal) or `intern-video2.5(multi-modal)` or '
          ' `internvl3(multi-modal)` or `qwenvl`(multi-modal), `qwen2vl`(multi-modal),'
          ' `deepseek-r1`(deepseek-r1 text llm), `qwq`(qwq text llm), `qwen3`(qwen3 text llm)'
          '`janus-pro`(multi-modal), `olm-ocr`(multi-modal).')
    exit(1)
demo.launch(server_name='0.0.0.0', server_port=SERVER_PORT)
