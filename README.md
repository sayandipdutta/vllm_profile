## Dependencies:

1. [uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)


## Setting Up:

```shell
git clone https://github.com/sayandipdutta/vllm_profile.git
cd vllm_profile
uv sync --no-group compile
uv sync --group compile
```


## Usage:

First copy an image (say, "pelican.png") into the working directory.

Create a profile directory in the working directory:

```shell
mkdir profiles
```

Start the server:

```shell
VLLM_TORCH_PROFILER_DIR=./profilers uv run python simple_server.py
```

From another shell, send a GET request via `curl` with your image as uri in the following format to the `/process` endpoint:

```shell
curl -X GET -H "Content-Type: application/json" -d '{"image": "file:///path/to/vllm_profile/pelican.png"}' http://localhost:8000/process
```

Considering `pwd` returns `/home/sayan/vllm_profile`, the request will look like:

```shell
curl -X GET -H "Content-Type: application/json" -d '{"image": "file:///home/sayan/vllm_profile/pelican.png"}' http://localhost:8000/process
```
