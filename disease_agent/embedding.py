# 임베딩 모듈 solar

import requests
from jihye.config import SOLAR_API_KEY

class Embedding:

    def __init__(self,
                 model: str = "solar-embedding-1-large-passage"):
        self.endpoint = "https://api.upstage.ai/v1/solar/embeddings"
        self.headers  = {
            "Authorization": f"Bearer {SOLAR_API_KEY}",
            "Content-Type":  "application/json"
        }
        # 사용할 임베딩 모델 지정
        self.model    = model

    def embed(self, text: str):

        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]):

        payload = {
            "input": texts,
            "model": self.model
        }
        resp = requests.post(self.endpoint,
                             json=payload,
                             headers=self.headers)
        resp.raise_for_status()
        data = resp.json()
        
        if "data" in data and isinstance(data["data"], list):
            return [item.get("embedding") for item in data["data"]]
        return data.get("embeddings", [])
