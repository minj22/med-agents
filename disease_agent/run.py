import pickle
import numpy as np
from rag import VectorDB
from agent.disease_classifier import DiseaseAgent

def main():
    # 1) VectorDB 로드
    db = VectorDB(dim=768, index_path="model/faiss.index")
    with open("model/faiss_texts.pkl", "rb") as f:
        db.texts = pickle.load(f)

    # 2) 에이전트 초기화
    agent = DiseaseAgent(db)

    # 3) 대화 루프
    print("=== 질병 에이전트 실행 ===")
    while True:
        symptom = input("[증상]: ").strip()
        if symptom.lower() in ("exit", "quit"):
            break

        # 분류 결과
        preds = agent.classify(symptom, top_k=3)
        diseases = [name for name, _ in preds]
        print(f"[Top-3 질병]: {diseases}\n")

        # RAG 검색: 각 질병별로 1개씩 가져오기
        snippets_per_disease = {}
        for d in diseases:
            emb_input = f"{symptom} {d}"
            q_emb = np.array([agent.emb.embed(emb_input)], dtype="float32")
            idxs, _ = db.search(q_emb, top_k=1)
            snippets_per_disease[d] = [db.texts[i] for i in idxs[0]]

        # 검색된 문서(스니펫) 출력
        print("[검색된 문서]:")
        for d in diseases:
            for doc in snippets_per_disease[d]:
                print(f"\n질병명: {d}")
                for line in doc.split("\n"):
                    print(line)
                print("---")
        print()

        # 최종 답변 생성
        answer = agent.answer(symptom, top_k=3)

        # 최종 답변 출력
        print("[최종 답변]:")
        print(answer)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()