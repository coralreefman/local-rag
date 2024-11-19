from sentence_transformers import SentenceTransformer

docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]

queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]

model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)

query_type = "s2p_query"
query_embeddings = model.encode(queries, prompt_name=query_type)
doc_embeddings = model.encode(docs)

similarities = model.similarity(query_embeddings, doc_embeddings)
print(similarities)

# for doc in docs:

#     mpnet_embeddings = get_sentence_transformer_embeddings(doc)

