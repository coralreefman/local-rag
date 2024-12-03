import json
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI

def prepare_qa_documents(file_path):
    with open(file_path, 'r') as f:
        qa_data = json.load(f)
    
    documents = [
        Document(
            page_content=item["answer"],
            metadata={"question": item["question"]}
        )
        for item in qa_data
    ]
    
    return documents

documents = prepare_qa_documents("../data/home0001qa.json")
print(documents[:5])


llm = ChatOpenAI(model="gpt-4o")

from langchain_nomic import NomicEmbeddings

model = NomicEmbeddings(model="")

from langchain_cohere import CohereEmbeddings
# needs API
cohere_embeddings = CohereEmbeddings(model="embed-english-light-v3.0")



from langchain_together.embeddings import TogetherEmbeddings

embed = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    # api_key="...",
    # other params...
)



from langchain_community.embeddings import GPT4AllEmbeddings
gpt4all_embd = GPT4AllEmbeddings() # POSSIBLE FOR NEW NOMIC MODEL?


# NEEDS API
from langchain_mistralai import MistralAIEmbeddings

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)