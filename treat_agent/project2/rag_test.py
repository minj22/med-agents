import torch
from config import PICKLE_PATH, DEVICE
from rag.retriever import retrieve_rag_data
from agent.jinryo_agent import JinryoAgent

def predict_and_test_rag_only(agent):
    while True:
        text = input("\n 증상 입력> ").strip()
        if not text:
            break

        preds = ["소화불량", "식도염", "급성 위장염"] 

        rag_qas_per_disease = []
        for disease in preds:
            qas = retrieve_rag_data(PICKLE_PATH, disease, text, top_k=3)
            rag_qas_per_disease.append(qas)

        print("\n RAG 검색 결과 (총 9개 Q-A):")
        for disease, qas in zip(preds, rag_qas_per_disease):
            print(f"\n [{disease}]")
            if not qas:
                print("  - 해당 질병에 대한 Q-A 없음")
                continue
            for i, qa in enumerate(qas):
                print(f"  Q{i+1}: {qa['question']}")
                print(f"  A{i+1}: {qa['answer']}\n")

        prompt = agent.build_prompt(text, preds, rag_qas_per_disease)
        print("\n 진료 에이전트 응답:")
        print(agent.generate_response(prompt))

def main():
    agent = JinryoAgent()
    predict_and_test_rag_only(agent)

if __name__ == "__main__":
    main()
