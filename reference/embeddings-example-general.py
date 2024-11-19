from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np

def get_sentence_transformer_embeddings(texts, model_name='all-mpnet-base-v2'):
    """    
    Args:
        texts: List of strings to embed
        model_name: Name of the model to use
    Returns:
        numpy array of embeddings
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.to(device)
    embeddings = model.encode(
        texts,
        device=device,
        convert_to_numpy=True
    )
    return embeddings

def get_huggingface_embeddings(texts, model_name='dunzhang/stella_en_400M_v5', pooling='mean'):
    """
    Generate embeddings using HuggingFace transformers with CUDA support.
    
    Args:
        texts: List of strings to embed
        model_name: Name of the model to use
        pooling: Pooling strategy ('mean' or 'cls')
    Returns:
        numpy array of embeddings
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and tokenizer with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    
    # Tokenize and move everything to GPU immediately
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        token_embeddings = model_output[0]  # First element contains token embeddings
        
        if pooling == 'cls':
            embeddings = token_embeddings[:, 0]
        else:  # mean pooling
            attention_mask = encoded_input['attention_mask']
            # Make sure mask is on the same device as token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().to(device)
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
    
        # Move to CPU before converting to numpy
        embeddings = embeddings.cpu().numpy()
    
    return embeddings

# Example usage
if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I love machine learning and natural language processing.",
        "This is a test sentence to demonstrate embeddings."
    ]
    
    # Using sentence-transformers
    st_embeddings = get_sentence_transformer_embeddings(texts)
    print("Sentence Transformers embedding shape:", st_embeddings.shape)
    
    # Using HuggingFace transformers directly (with mean pooling)
    hf_embeddings = get_huggingface_embeddings(texts, pooling='mean')
    print("HuggingFace embedding shape:", hf_embeddings.shape)

    # Calculate similarity on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor1 = torch.tensor(st_embeddings[0].reshape(1, -1)).to(device)
    tensor2 = torch.tensor(st_embeddings[1].reshape(1, -1)).to(device)
    similarity = F.cosine_similarity(tensor1, tensor2)
    print(f"\nCosine similarity between first two sentences: {similarity.cpu().item():.4f}")