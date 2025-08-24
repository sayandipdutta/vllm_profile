from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import os
import threading
import json
import queue
import time
import uuid
from vllm import LLM
from pathlib import Path

request_queue = queue.Queue()
results = {}
llm = LLM(
    model="hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4",
    allowed_local_media_path=str(Path(__file__).parent),
)

# os.environ["VLLM_TORCH_PROFILER_DIR"] = "./profiles"


def worker():
    while True:
        request_data = request_queue.get()
        if request_data is None:
            break

        print(f"Processing request: {request_data['data']}")
        data = json.loads(request_data["data"])
        image_url = data["image"]
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What's in this image?"},
                ],
            },
        ]

        llm.start_profile()
        outputs = llm.chat(conversation)
        llm.stop_profile()
        for o in outputs:
            generated_text = o.outputs[0].text
            results[request_data["id"]] = generated_text

        request_queue.task_done()


# file:///home/sayan/projects/vllmdemo/CO2898W1989_00068-Others.TIF

num_worker_threads = 1
threads = []
for _ in range(num_worker_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Server is running. Send POST requests to /process")
        elif self.path == "/process":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)

            req_id = str(uuid.uuid4())
            request_queue.put({"id": req_id, "data": post_data.decode("utf-8")})
            while req_id not in results:
                print("Waiting")
                time.sleep(5)
            result = results[req_id]
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(str(result).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    host = ""
    port = 8000
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, RequestHandler)
    print(f"Starting server on {host}:{port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(num_worker_threads):
            request_queue.put(None)
        for t in threads:
            t.join()
        httpd.server_close()
        print("Server stopped.")
