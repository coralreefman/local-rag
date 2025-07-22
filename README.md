# local-rag

## Setting up the environment (Linux)
```bash
# install basics: Cuda, torch, transformers
conda install nvidia/label/cuda-12.4.0::cuda-toolkit

conda install nvidia/label/cuda-12.4.0::libcublas-dev

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformers 

pip install sentence-transformers

# langchain libs
pip install langchain langchain-chroma langchain_community langchainhub langgraph

pip install xformers

# optional: for libs that require compilation, use this instead of system cmake
conda install -c conda-forge cmake libstdcxx-ng
```

## Local Runtimes


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

## Hosted LLM/RAG platforms

**Together.ai**
- seems to be quite rudimentary, at least their cookbooks look much less sophisticated than what’s possible w/ Langchain — it still looks promising for fine-tuning though.
- offers serverless/dedicated GPU endpoints
- supports LlamaIndex integration and multi-model workflows

**AWS bedrock (Bedrock + AgentCore)**
- UI/wrapper/Cloud Compute, but nothing fancy that can’t be done locally. Mainly worth it for easy deployment as it seems.
- basically hosted inference + RAG via Knowledge Bases, also offering 'semantic agent service'.
- S3 Vectors: native vector storage integrated with Knowledge Bases
- supports Claude 4, Claude 3.5, Llama 4 Scout/Maverick, Nova Premier/Canvas, custom Nova on-demand

**Cohere**
- provides LLM + embeddings via API.
- Example integrations use their own LangChain/Hugging Face wrappers and RAG pipelines.
- uses llama index and langchain in their examples. they provide an LLM through their Langchain package like `llm = ChatCohere(model="command-r")` and some rerank shenanigans for RAG.
https://docs.cohere.com/page/creating-a-qa-bot

**Contextual ai**
- enterprise-grade RAG/agent platform built around custom GLM and instruction models 
- offers reranking, retrieval, grounding, and memory via an agent loop
- focused on compliance for corporate search/Q&A pipelines

**Google (Vertex AI)**
- provides hosted Gemini models (1.5 Flash, 1.5 Pro) with long context, multimodal input, and tool use
- includes embedding models and native doc ingestion
- Vertex AI Search: full RAG pipeline with chunking, vectorization, retrieval, reranking, grounding, citation
- designed for enterprise-scale QA/search systems without needing LangChain or LlamaIndex
https://cloud.google.com/vertex-ai/docs/generative-ai/overview


## Frameworks

### Langchain

https://python.langchain.com/docs/tutorials/chatbot/

https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/

https://www.youtube.com/watch?v=bq1Plo2RhYI

https://blog.langchain.dev/self-learning-gpts/

https://python.langchain.com/docs/concepts/retrieval/

### Sentence Transformer

https://sbert.net/

- Python module for accessing, using, and training state-of-the-art text and image embedding models

- useful for [semantic search](https://sbert.net/examples/applications/semantic-search/README.html), [semantic textual similarity](https://sbert.net/docs/usage/semantic_textual_similarity.html), and [paraphrase mining](https://sbert.net/examples/applications/paraphrase-mining/README.html).

- super simple to use, much easier than regular transformers

- Use sentence-transformers for production/simple cases, transformers for research/custom needs.

Usage example:  
```python
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```

**For speed up in PyTorch use fp16**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences)
```