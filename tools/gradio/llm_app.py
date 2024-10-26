import gradio as gr
import sys
import openai

if len(sys.argv) < 3:
    print('python3 llm_app.py <app_type> <llm_server>')
    exit(1)

app_type = sys.argv[1]
if app_type not in ['llm', 'internvl2']:
    print('`app_type` only support `llm`(all text llm) or `internvl2`(multi-modal)')
    exit(1)

llm_server = sys.argv[2]


def llm_fn(message, history):
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
        stream=True
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


def multi_modal_llm_fn(message, history):
    # print('message:', message)
    # print('history:', history)

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
            image_flag_count = pre_messages[-1].count('<image>')
            if image_flag_count == 0:
                # insert <image> in text start
                if len(pre_messages[:-1]) == 1:
                    msg['content'][0]['text'] = '<image>' + pre_messages[-1]
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
            new_message['content'][0]['text'] = '<image>' + message['text']
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

    content = ''
    try:
        for msg in res:
            # print('msg:', msg)
            if msg.choices[0].delta.content is not None:
                content += msg.choices[0].delta.content
                yield content
    except openai.APIError as e:
        print('error:', e)
        if '[TrtInfererException] Dims not match' in e.message:
            yield 'error: 图片尺寸过大或超过图片个数限制。'
        else:
            yield 'error: ' + e.message.replace('<image>', '`<image>`')
    except Exception as e:
        print('error:', e)
        yield 'error: ' + str(e)


if app_type == 'llm':
    demo = gr.ChatInterface(fn=llm_fn, type="messages", examples=[
        "你好，你是谁？",
        "提供一段快速排序的c++代码：",
        "中国长城有多长？",
        "世界上第一高峰是哪座山？",
    ], title="grps-trtllm", multimodal=False)
else:
    demo = gr.ChatInterface(fn=multi_modal_llm_fn, type="messages", examples=[
        {"text": "你好，你是谁？"},
        {"text": "描述一下这张图片：",
         "files": ['https://i2.hdslb.com/bfs/archive/7172d7a46e2703e0bd5eabda22f8d8ac70025c76.jpg']},
        {"text": "描述一下这两张图片：",
         "files": [
             'https://p6.itc.cn/q_70/images03/20230821/69b103277521450e89090a24df1327d7.jpeg',
             'https://i0.hdslb.com/bfs/archive/dd8dfe1126b847e00573dbda617180da77a38a06.jpg']}],
                            title="grps-trtllm",
                            multimodal=True)

demo.launch(server_name='0.0.0.0')
