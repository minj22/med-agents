import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

from config import *
from data.loader import load_records, prepare_data
from model.classifier import TextDataset, build_model, save_model, load_model
from rag.retriever import retrieve_rag_data
from agent.jinryo_agent import JinryoAgent

PICKLE_PATH = "rag_embeddings.pkl"

def predict_and_chat(model, tokenizer, id2label, agent):
    model.eval()
    while True:
        text = input("\n증상 입력> ").strip()
        if not text:
            break

        toks = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        ).to(DEVICE)

        logits = model(**toks).logits
        top3_indices = logits.topk(3, dim=-1).indices.squeeze(0).tolist()
        preds = [id2label[i] for i in top3_indices]

        rag_qas_per_disease = []
        for disease in preds:
            qas = retrieve_rag_data(PICKLE_PATH, disease, text, top_k=3)
            rag_qas_per_disease.append(qas)

        print("\n 질병별 유사 Q-A 3개씩 출력:")
        for disease, qas in zip(preds, rag_qas_per_disease):
            print(f"\n [{disease}]")
        if not qas:
            print("  - 해당 질병에 대한 Q-A 없음")
            continue

        for i, qa in enumerate(qas):
            print(f"  Q{i+1}: {qa['question']}")
            print(f"  A{i+1}: {qa['answer']}\n")


        prompt = agent.build_prompt(text, preds, rag_qas_per_disease)
        print("\n진료 에이전트 응답:")
        print(agent.generate_response(prompt))

def main():
    torch.manual_seed(SEED)

    recs = load_records(JSON_DIR)
    _, _, label2id, id2label = prepare_data(recs, SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(MODEL_NAME, num_labels=len(label2id)).to(DEVICE)

    model = load_model(model, MODEL_PATH).to(DEVICE)
    print("\n모델 로딩 완료. 대화 모드 진입.")

    agent = JinryoAgent()

    predict_and_chat(model, tokenizer, id2label, agent)


if __name__ == "__main__":
    main()
