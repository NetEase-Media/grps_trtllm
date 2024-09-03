import random
import sys
import time
import requests
import threading

latency_list = []
tokens_list = []
speed_list = []

short_fixed_prompt = '华盛顿是谁？'

long_fixed_promt = ('****请用大概200字左右简述下面新闻****\n \
今年1月，腾讯2023年会现场，马化腾用35分钟公开“反思”了腾讯过去这一年。马化腾提到，过去一年的表现虽称不上完美无缺，但腾讯确实走出了一部分“\
至暗时刻”。从财报看，腾讯似乎已经度过了寒冬，迅速回暖，开始努力赚钱。8月14日，腾讯发布财报，今年上半年，腾讯控股实现营收3206.18亿元，\
同比增长7%；归母净利润895.19亿元，同比增长72％，相当于每天净赚4.92亿元（2024年上半年共计182天，数据仅供参考）。\
日赚一个小目标，马化腾，似乎可以松一口气了。腾讯这一次的财报，发布得很低调。上一次发财报时，腾讯还在强调V曲线，这一次，马化腾则开始谈创造新的\
商业价值和更好地服务用户需求。从营收和净利润看，腾讯在上半年和Q2的表现是非常亮眼的。2024年第二季度，腾讯控股营收1611.2亿元，上年同期\
1492.08亿元，同比增长8%；公司权益持有人应占盈利为476.3亿元，同比增长82%；调整后净利润573.1亿元，同比增长53%；毛利859亿元，同比增长21%。\
如果以年为区间，腾讯的净利润的上涨则更为明显。2023年二季度至2024年二季度，腾讯5份财报的非国际财报准则下净利润同比增幅分别是33%、39%、44%、\
54%、53%。连续五个财季实现上涨，腾讯这一次，真的赚麻了。与此同时，腾讯年初定下的千亿港元回购目标，如今已完成523亿港元，连续两个季度稳居港股\
回购王，且回购力度仍在继续加大。每天净赚4.92亿元，又在资本市场上不断回购，腾讯二级市场的表现也一路高歌，截至14日收盘，腾讯市值已经高达3.48万\
亿港元。净利润保持超过50%的增长，腾讯，哪些业务在猛赚钱？按照业务营收贡献占比，增值服务49%、金融科技及企业服务31%、网络广告19%。而所谓增值\
服务，就是游戏业务。上半年，游戏业务带来的营收同比增长6%至788.22亿元。今年上半年，腾讯国际市场游戏营收增长至139亿元，本土市场游戏营收增长至\
346亿元。根据《2024H1全球移动游戏市场数据报告》显示，今年上半年中国移动游戏App Store 收入TOP10中，腾讯旗下游戏占据6席。总被吐槽的腾讯游戏\
，终于扭转了之前的萎靡状态，扬眉吐气了一次。除了游戏之外，广告业务收入同比增长高达19%，是营收贡献中，增长最快的一部分。而这一业务的增长，则来\
源自视频号和长视频。年初马化腾讲话时，特别强调了微信视频号的成功。他称视频号不仅是在腾讯短视频失利的情况下提供了新的可能性。这一成果，并不是\
简单地跟风，而是结合了微信强大的社交生态，走出了一条独特的发展之路。这次财报中，腾讯提到了微信及WeChat的合并月活跃账户数增至13.71亿。更多活\
跃用户的带动下，微信小程序用户时长同比增长超20%，通过小程序促成的交易额实现同比双位数增长。视频号总用户使用时长同比显著增长。从微信公开课2024\
的数据来看，视频号直播带货的GMV（成交总额）达到了2022年的3倍，供给数量增长了300%，订单数量增长了244%，GPM（每千观众下单的总金额）超过了900\
元。这些数据不仅展示了视频号的市场潜力，也体现了腾讯在视频号上的实力。上半年，金融科技及企业服务营收增至504.40亿，网络广告则入账297.71亿，\
进一步提振了腾讯整体营收情况。有了这三驾马车，马化腾，这一次，可以松一口气了。从2022年起，腾讯开始降本增效，其中减员也是其中一步。过去两年，\
腾讯共计减掉了近万人。尤其是在2022年第二季度，由于当时净利润下降了56%，腾讯此后仅仅用了3个月时间（同年3月、6月相比），就将员工减少了5500人\
左右。马化腾年初曾提出，腾讯在管理上要重视小团队精神，要求管理者从招聘源头就慎重，尽量避免大规模招人后又快速大规模裁员，所以“没想清楚的时候小\
团队先试，不要急”。老板一发话，果然招聘团队也积极响应。今年一季度，腾讯的招聘显得很保守，轻微裁员了：2023年Q1为106221人，2024年Q1变为\
104787人，同比减少1434人（1.36%）。要知道，今年Q1时，已经是腾讯连续第六个季度人员减少。而根据二季度财报显示，截至2024年6月30日，腾讯\
有105506名员工，比半年前多了89人。腾讯终于结束大规模优化了。不再规模化裁员，第一点说明腾讯裁员不是一味省钱，第二点反映出即便业务好了，腾讯\
开始回暖，赚钱了。')

random_prompts = [
    '今天天气真好，适合出门散步',
    '中国的长城有多长？',
    '人工智能对未来社会有什么影响？',
    '传统文化在现代社会中的地位如何？',
]


def request(server, prompt):
    if type(prompt) is list:
        prompt_idx = random.randint(0, len(prompt) - 1)
        text_inp = prompt[prompt_idx]
    else:
        text_inp = prompt

    url = f'{server}/v1/chat/completions'
    data = {
        "model": "",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": text_inp
            }
        ],
        "max_tokens": 512,
        "top_p": 0.3,
        "repetition_penalty": 1.0,
        "temperature": 0.1
    }
    headers = {'Content-Type': 'application/json'}
    start = time.time()
    response = requests.post(url, json=data, headers=headers).json()
    end = time.time()
    latency = (end - start) * 1000
    text_output = response['choices'][0]['message']['content']
    print(text_output, flush=True)
    input_token_len = response['usage']['prompt_tokens']
    output_token_len = response['usage']['completion_tokens']
    total_tokens = response['usage']['total_tokens']
    speed = total_tokens / latency * 1000
    print(f'Latency: {latency} ms', flush=True)
    print(f'Input tokens: {input_token_len}, Output tokens: {output_token_len}, Total tokens: {total_tokens}',
          flush=True)
    print(f'Speed: {speed} tokens/s', flush=True)
    latency_list.append(latency)
    tokens_list.append(total_tokens)
    speed_list.append(speed)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python openai_benchmark.py <server> <concurrency> <prompt_type>")
        exit(1)
    server = sys.argv[1]
    concurrency = sys.argv[2]
    prompt_type = sys.argv[3]


    def run(server, prompt):
        for _ in range(10):
            request(server, prompt)


    th = []
    for i in range(int(concurrency)):
        if prompt_type == '0':
            prompt = short_fixed_prompt
        elif prompt_type == '1':
            prompt = long_fixed_promt
        elif prompt_type == '2':
            prompt = random_prompts

        t = threading.Thread(target=run, args=(f'http://{server}', prompt))
        th.append(t)
        t.start()
    for t in th:
        t.join()

    print(f'Average Latency: {sum(latency_list) / len(latency_list)} ms', flush=True)
    print(f'Average Tokens: {sum(tokens_list) / len(tokens_list)}', flush=True)
    print(f'Average Speed: {sum(speed_list) / len(speed_list) * int(concurrency)} tokens/s', flush=True)
