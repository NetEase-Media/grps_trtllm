import random
import sys
import time
import requests
import threading
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/tmp/Qwen2-7B')

latency_list = []
tokens_list = []
speed_list = []

short_fixed_prompt = '华盛顿是谁？'

long_fixed_promt = ('****请总结下面新闻，总结字数大概200字左右****\n \
8月15日消息，谷歌前首席执行官兼执行董事长埃里克·施密特（Eric Schmidt）此前表示，谷歌在人工智能竞赛中落后都怪远程办公政策。\
但周三他收回了这番言论。施密特在周三的一封电子邮件中表示：“我在提及谷歌及其工作时间时说错了话，对此我感到遗憾。\
”施密特于五年前离开谷歌的母公司Alphabet。他此前在斯坦福大学的一次公开演讲时，针对有关谷歌如何与OpenAI竞争的问题，批评了谷歌的远程工作政策。\
在斯坦福大学的演讲中，施密特说：“谷歌决定更重视工作与生活的平衡、早点回家和在家工作，而不是赢得竞争。”他补充说：“之所以创业公司能成功，\
是因为员工在拼命工作。”本周，提供在线课程的斯坦福在线(Stanford Online)将施密特的演讲视频发布在YouTube上。截至周三下午，\
该视频的观看次数已超过40,000。后来，这个视频被设为了私密状态。施密特称，他要求将该视频撤下，并拒绝进一步评论。斯坦福大学未对有关视频的评论\
请求作出回应。谷歌和OpenAI自新冠大流行以来都实施了类似的复工政策。自2022年起，两家公司都要求员工每周至少来办公室工作三天。谷歌在周三强调了\
混合工作模式的好处。该公司表示，会联系那些未能满足每周三天到办公室要求的员工，提醒他们必须到岗办公的规定。除了施密特之外，包括摩根大通首席\
执行官杰米·戴蒙（Jamie Dimon）和特斯拉首席执行官埃隆·马斯克（Elon Musk）等许多企业高管都曾对居家办公政策表示不满，称这些政策使公司效率\
降低，竞争力减弱。戴蒙几年前在一封年度信中说，高层领导不能只是坐在办公桌后或屏幕前。马斯克则表示，员工每周至少需要在办公室工作40小时。\
代表美国和加拿大超过1,000名员工的Alphabet工人工会在X上发布帖子称：“灵活的工作安排并没有减慢谷歌员工的工作进度。缺乏人手、优先事项的\
频繁变动、持续的裁员、工资停滞不前以及管理层在项目跟进上的不足，这些才是每天拖慢我们步伐的真正原因。”根据Alphabet的年度报告，截至去年底，\
该公司共有约182,000名员工。公司有时难以使员工重返办公室，一些员工抱怨通勤时间长和照顾家人的责任。在某些情况下，员工已对这些要求提出反对。\
施密特对学生表示，在竞争激烈的创业环境中，办公室工作是成功的必要条件。施密特当时说：“如果你们毕业后开公司，想和其他初创公司竞争的话，\
你们就不会让员工在家工作，每周只来办公室一天。”施密特从2001年到2011年担任谷歌的首席执行官，并于2018年卸任执行董事长，2019年离开Alphabet\
董事会。据FactSet的数据，他仍是Alphabet的股东。他与妻子共同创立了施密特未来基金会，该基金会资助科学技术研究。他还是“特别竞争研究项目”\
（Special Competitive Studies Project）的主席，这是一个专注于美国的人工智能和其他技术的非营利组织。自OpenAI在2022年底推出ChatGPT\
以来，谷歌一直在人工智能领域保持防御态势。今年早些时候，该公司推出的Gemini聊天机器人因被指存在偏见而受到批评。公司已加强了Gemini的功能，\
并将其提供在公司的四款新Pixel手机上。其主要特点是改进了语音助手，具备更自然的对话能力。\n \
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
    if type[prompt] == 'list':
        prompt_idx = random.randint(0, len(prompt) - 1)
        text_inp = prompt[prompt_idx]
    else:
        text_inp = prompt

    text_inp = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + text_inp + "<|im_end|>\n<|im_start|>assistant"
    url = f'{server}/v2/models/ensemble/generate'
    data = {
        "text_input": text_inp,
        "max_tokens": 1024,
        "bad_words": "",
        "stop_words": ["<|im_start|>", "<|im_end|>"],
        "end_id": 151643,
        "pad_id": 151643,
        "top_p": 0.3,
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        "temperature": 0.1,
        "stream": False
    }
    headers = {'Content-Type': 'application/json'}
    start = time.time()
    response = requests.post(url, json=data, headers=headers).json()
    end = time.time()
    latency = (end - start) * 1000
    text_output = response['text_output']
    print(text_output, flush=True)
    input_token_len = len(tokenizer(text_inp)['input_ids'])
    output_token_len = len(tokenizer(text_output)['input_ids'])
    total_tokens = input_token_len + output_token_len
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
        print("Usage: python triton_benchmark.py <server> <concurrency> <prompt_type>")
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
