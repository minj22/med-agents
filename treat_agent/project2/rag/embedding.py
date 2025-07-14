import os, json, sys
import pickle
from langchain_upstage import UpstageEmbeddings

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import UPSTAGE_API_KEY

def collect_diagnosis_qas(json_dir):
    qas = []
    pattern = os.path.abspath(json_dir)

    for root, dirs, files in os.walk(pattern):
        if os.path.basename(root) != "진단":
            continue
        print(f"[DEBUG] 진단 폴더 탐색 중: {root} (파일 {len(files)}개)")
        for file in files:
            if not file.endswith(".json"):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, encoding="utf-8") as f:
                    j = json.load(f)
                question = j.get("question", "").strip()
                answer = j.get("answer", "").strip()
                if question and answer:
                    disease_name = os.path.basename(os.path.dirname(root))
                    qas.append({
                        "disease": disease_name,
                        "question": question,
                        "answer": answer,
                        "text": f"Q: {question}\nA: {answer}"
                    })
                    if len(qas) % 1000 == 0:  
                        print(f"[DEBUG] 수집된 Q-A 개수: {len(qas)}")
            except Exception as e:
                print(f"[WARNING] {path} 파일 로딩 실패: {e}")
                continue
    return qas


def embed_qas(qas):
    embedder = UpstageEmbeddings(
        api_key=UPSTAGE_API_KEY,
        model="solar-embedding-1-large-passage"
    )
    texts = [item["text"] for item in qas]
    embeddings = embedder.embed_documents(texts)
    return embeddings

def main():
    json_dir = sys.argv[1]
    qas = collect_diagnosis_qas(json_dir)
    print(f"[INFO] 총 {len(qas)}개의 진단 Q-A를 수집했습니다.")

    embeddings = embed_qas(qas)

    output = {
        "qas": qas,
        "embeddings": embeddings
    }

    with open("rag_embeddings.pkl", "wb") as f:
        pickle.dump(output, f)

    print("[INFO] rag_embeddings.pkl 저장 완료!")

if __name__ == "__main__":
    main()
