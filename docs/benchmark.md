# benchmark

提供一个简易的[压测脚本](../client/openai_benchmark.py)用于对服务性能进行压测，可以进行```非streaming```和```streaming```
两种压测。

## 压测脚本

```text
Usage: python openai_benchmark.py <server> <concurrency> <prompt_type> <max_tokens> <streaming> [tokenizer_path]

server: 服务地址
concurrency: 并发请求数
prompt_type: 0-固定短输入，1-固定长输入，2-随机短输入
max_tokens: 最大token数，用于固定输出长度
streaming: 是否使用streaming模式，true/false，非streaming模式统计简单的请求延迟和吞吐量，streaming模式统计首token延长以及后续token延迟和后续吞吐量。
tokenizer_path: tokenizer路径，streaming模式需要指定。
```

## 非streaming模式

```bash
# 4并发压测，限制最大token数为50，使用固定短输入
python3 client/openai_benchmark.py 0.0.0.0:9997 4 0 50 false

# 结果如下
: '
Latency: 2575.7412910461426 ms, Total tokens: 74, Speed: 28.72959340180657 tokens/s
Latency: 2575.5906105041504 ms, Total tokens: 74, Speed: 28.73127417773709 tokens/s
Latency: 2575.0246047973633 ms, Total tokens: 74, Speed: 28.73758948249867 tokens/s
Latency: 2574.532985687256 ms, Total tokens: 74, Speed: 28.74307705956471 tokens/s
--------------------------------------------------
Average Latency: 2571.776455640793 ms
Average Tokens: 74.0
Throughput: 115.09695530244566 tokens/s
'
```

## streaming模式

```bash
# 4并发压测，限制最大token数为50，使用固定短输入
python3 client/openai_benchmark.py 0.0.0.0:9997 4 0 50 true /tmp/QwQ-32B-AWQ/

# 结果如下
: '
First token latency: 93.83893013000488 ms, Follow tokens: 49, Follow latency: 2477.2791862487793 ms, Follow Speed: 19.779764942117108 tokens/s
First token latency: 179.8086166381836 ms, Follow tokens: 49, Follow latency: 2390.686511993408 ms, Follow Speed: 20.496204648405655 tokens/s
First token latency: 179.40092086791992 ms, Follow tokens: 49, Follow latency: 2390.542984008789 ms, Follow Speed: 20.497435238679586 tokens/s
First token latency: 180.94754219055176 ms, Follow tokens: 49, Follow latency: 2390.345811843872 ms, Follow Speed: 20.499126008132787 tokens/s
--------------------------------------------------
Average First token latency: 156.15192651748657 ms
Average follow latency: 2407.7471911907196 ms
Average follow tokens: 49.0
Follow throughput: 81.42325846430592 tokens/s
'
```