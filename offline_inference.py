from pathlib import Path
import os

from vllm import LLM
from vllm.assets.image import ImageAsset

from torch.profiler import profile, ProfilerActivity, record_function

import numpy as np
from PIL import Image

os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

image_path1 = Path(__file__).parent / "no_error.jpg"
image_path2 = Path(__file__).parent / "this errors.jpg"

Image.fromarray(np.random.randint(0, 256, (224, 224), dtype=np.uint8)).save(image_path1)
Image.fromarray(np.random.randint(0, 256, (224, 224), dtype=np.uint8)).save(image_path2)
assert image_path1.is_file() and image_path2.is_file()

llm = LLM(
    model="hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4",
    allowed_local_media_path=str(Path(__file__).parent),
)
image_url1 = image_path1.as_uri()
image_url2 = image_path2.as_uri()

print(image_path1, image_url1)
print(image_path2, image_url2)

conversation = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url1}},
            {"type": "image_url", "image_url": {"url": image_url2}},
            {"type": "text", "text": "What's in these images?"},
        ],
    },
]

# Perform inference and log output.
llm.start_profile()
outputs = llm.chat(conversation)
llm.stop_profile()
for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
