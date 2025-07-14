# 벡터 DB 생성·검색

import faiss
import numpy as np

class VectorDB:

    def __init__(self, dim: int, index_path: str = None):
        self.dim = dim
        if index_path:
            # 저장된 index 파일(.index) 로드
            cpu_index = faiss.read_index(index_path)
        else:
            # 빈 인덱스 생성
            cpu_index = faiss.IndexFlatIP(dim)

        # GPU 리소스 초기화
        self.res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(self.res, 0, cpu_index)

        # (선택) 색인과 대응되는 원문 텍스트 리스트
        self.texts = []

    def build(self, embeddings: np.ndarray, texts: list = None):

        self.index.add(embeddings)
        if texts is not None:
            self.texts = texts

    def search(self, query_emb: np.ndarray, top_k: int = 3):

        distances, indices = self.index.search(query_emb, top_k)
        return indices, distances
