# local-rag

## Runtimes


### Ollama

- wraps llama.cpp with model management, REST API, OpenAI compatibility
- basically just a wrapper around llama.cpp
- doesn't let you do all the things llama.cpp does
- it’s just a REST API service, and doesn't come with any UI apart from the CLI command, so you need to find your own UI for it (open-webui, OllamaChat, ChatBox etc.)
- it can also present an openai api and make all the local LLMs available via that api, so any tools that can talk to chatgpt can also connect to ollama.
- it exposes an OpenAI-like interface on localhost:11434/v1/chat/completions etc., which works with LangChain, LlamaIndex, etc.

to use GGUF files in ollama:

https://github.com/ollama/ollama/blob/main/docs/import.md

https://github.com/ollama/ollama/blob/main/docs/modelfile.md

commands:  
`systemctl stop ollama`

`ollama serve`

**Ollama Python bindings**

https://github.com/ollama/ollama-python


### Llama.cpp

- llama.cpp seems to be more versatile than ollama
- seems to only work with models in the GGUF format

**How to install Llama.cpp with python wrapper**

```bash
conda install nvidia/label/cuda-12.4.0::cuda-toolkit

conda install nvidia/label/cuda-12.4.0::libcublas-dev

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# for compilation, use this instead of system cmake
conda install -c conda-forge cmake libstdcxx-ng

MAKEFLAGS="-j$(nproc)" CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose

# for some reason i'm getting a segmentation fault CORE DUMPED when 
# using the wrapper installed above with a meta model i converted myself
# but it still generates an answer SOLVED n_gpu_layers was set to -33
```

**Install without python wrapper for CLI**

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# for some reason llama.cpp compilation can't find cublas, 
# creating a symbolic link works
sudo ln -s /home/studio/anaconda3/envs/llm/include/cublas_v2.h /home/studio/anaconda3/envs/llm/targets/x86_64-linux/include/cublas_v2.h

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 24
cmake --install build --prefix $CONDA_PREFIX

# test install
llama-cli -m /media/studio/working-01/GGUF-Models/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-D_AU-Q8_0.gguf -p "i am" -ngl 32 --temp 1.5 --repeat-penalty 1.5 -n 128

# to install python wrapper without recompiling llama.cpp
# DOES NOT WORK AS EXPECTED
export LLAMA_CPP_LIB=$CONDA_PREFIX/lib/libllama.so
CMAKE_ARGS="-DLLAMA_BUILD=OFF" pip install llama-cpp-python
```

Usage:

https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md

Python Wrapper:

https://github.com/abetlen/llama-cpp-python