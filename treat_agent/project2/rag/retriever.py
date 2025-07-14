import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings

import os


class SolarUpstageEmbeddings:
    def __init__(self, api_key=os.getenv("UPSTAGE_KEY_3"), model="solar-embedding-1-large-passage"):
        self.client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
        self.model = model

    def embed_query(self, text: str):
        if not text.strip():
            return [0.0] * 1024
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

def retrieve_rag_data(pickle_path, disease_name, query, top_k=3):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    qas = data["qas"]
    embeddings = np.array(data["embeddings"])

    
    embed_key = os.getenv("EMBEDDING_KEY")
    filtered_qas = [qa for qa in qas if qa["disease"].lower() == disease_name.lower()]
    if not filtered_qas:
        return []

    filtered_indices = [i for i, qa in enumerate(qas) if qa["disease"].lower() == disease_name.lower()]
    filtered_embeddings = embeddings[filtered_indices]

    embedder = UpstageEmbeddings(api_key=embed_key, model="embedding-query")

    query_vec = np.array(embedder.embed_query(query)).reshape(1, -1)

    sims = cosine_similarity(query_vec, filtered_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    result_qas = [filtered_qas[i] for i in top_indices]
    return result_qas