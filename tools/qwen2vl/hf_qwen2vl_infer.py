import sys
import time

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

path = sys.argv[1]

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    path, torch_dtype="float16", device_map="cuda"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
min_pixels = 4 * 28 * 28
max_pixels = 1024 * 1024 / 4
processor = AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file://./data/image1.jpg",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            {
                "type": "image",
                "image": "file://./data/image2.jpg",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            {"type": "text", "text": "描述一下两张图片的不同。"},
        ],
    }
]

# Preparation for inference
begin = time.time()
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print('text:', text)
image_inputs, video_inputs = process_vision_info(messages)
print('image_inputs: {}, video_inputs: {}'.format(image_inputs, video_inputs))
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
pixel_values = inputs['pixel_values']
image_grid_thw = inputs['image_grid_thw']
print('input_ids shape: {}, input_ids: {}'.format(input_ids.shape, input_ids.tolist()))
print('attention_mask shape: {}, attention_mask: {}'.format(attention_mask.shape, attention_mask))
print('pixel_values shape: {}, pixel_values: {}'.format(pixel_values.shape, pixel_values))
print('image_grid_thw shape: {}, image_grid_thw: {}'.format(image_grid_thw.shape, image_grid_thw))

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
print("Time taken: ", time.time() - begin)
