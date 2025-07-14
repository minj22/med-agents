import os
import json
import pickle
import numpy as np
import faiss

from embedding import Embedding
from glob import glob

from config import SOLAR_API_KEY
print(f"[DEBUG] SOLAR_API_KEY={SOLAR_API_KEY!r}")

# 1) JSON 파일 경로 (answered_output 하위 전체 탐색)
JSON_ROOT = "answered_output"

# 2) 텍스트 수집: question 필드만 사용
texts = []
for root, _, files in os.walk(JSON_ROOT):
    for fn in files:
        if not fn.endswith(".json"):
            continue
        path = os.path.join(root, fn)
        data = json.load(open(path, "r", encoding="utf-8"))
        # 의도(intention)가 '정의'인 경우에만 처리
        if data.get("intention") != "정의":
            continue

        # disease_name, question, answer 을 한 문자열로 합치기
        parts = []
        dn = data.get("disease_name", {}).get("kor")
        if dn:
            parts.append(f"[질병명] {dn}")
        q = data.get("question")
        if q:
            parts.append(f"[질문] {q}")
        a = data.get("answer")
        if a:
            parts.append(f"[답변] {a}")

        full_text = "\n".join(parts)
        texts.append(full_text)

# 3) Solar 임베딩 생성
emb = Embedding()

emb_list = []
BATCH   = 64
emb_list = []
total    = len(texts)

for start in range(0, total, BATCH):
    batch_texts = texts[start : start + BATCH]
    batch_embs  = emb.embed_batch(batch_texts)
    emb_list.extend(batch_embs)
    done = min(start + BATCH, total)
    print(f"[Progress] {done}/{total} embeddings generated")
embs = np.array(emb_list, dtype="float32")

# 4) FAISS 인덱스 빌드 및 저장
index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
os.makedirs("model", exist_ok=True)
faiss.write_index(index, "model/faiss.index")

# 5) 텍스트 리스트도 함께 저장
with open("model/faiss_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Vector DB 생성 완료: model/faiss.index, model/faiss_texts.pkl")