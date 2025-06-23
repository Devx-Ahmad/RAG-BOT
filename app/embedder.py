# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def build_index(chunks):
#     embeddings = model.encode(chunks)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(embeddings)
#     return index, embeddings

# def search(query, index, chunks, top_k=5):
#     query_embedding = model.encode([query])
#     D, I = index.search(np.array(query_embedding), top_k)
#     return [chunks[i] for i in I[0]]
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks):
    # Normalize chunk embeddings for cosine similarity
    embeddings = model.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Use inner product
    index.add(embeddings)
    return index, embeddings

def search(query, index, chunks, top_k=5):
    #normalize the query as well
    query_embedding = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]
