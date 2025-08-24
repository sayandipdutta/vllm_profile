from openai import OpenAI
from torch.profiler import profile, ProfilerActivity, record_function

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Single-image input inference
image_url = "file:///home/sayan/projects/argo/image1.TIF"
image_url = "file:///home/sayan/projects/argo/image2.TIF"
prompt = """
Only extract the requested fields, and do not output any bounding boxes. The response \
should contain only the requested fields as keys, and their values as values.
"""

import time

start = time.perf_counter()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    with record_function("model_inference"):
        chat_response = client.chat.completions.create(
            model="hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4",
            messages=[
                {
                    "role": "user",
                    "content": [
                        # NOTE: The prompt formatting with the image token `<image>` is not needed
                        # since the prompt will be processed automatically by the API server.
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )
    print(prof.key_averages().table())
end = time.perf_counter()
print("Chat completion output:", chat_response.choices[0].message.content)
print(f"Took: {end - start} seconds")

# Multi-image input inference
# image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
# image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"
#
# chat_response = client.chat.completions.create(
#     model="microsoft/Phi-3.5-vision-instruct",
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What are the animals in these images?"},
#                 {"type": "image_url", "image_url": {"url": image_url_duck}},
#                 {"type": "image_url", "image_url": {"url": image_url_lion}},
#             ],
#         }
#     ],
# )
# print("Chat completion output:", chat_response.choices[0].message.content)
