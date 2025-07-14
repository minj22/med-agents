# 질병 분류기 Agent -> 여기에 훈련된 분류기 로드

import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from jihye.config import DEVICE
from jihye.embedding import Embedding
from jihye.rag import VectorDB
from jihye.llm import LLM

class DiseaseAgent:

    def __init__(
        self,
        db: VectorDB,
        model_path: str = "jihye/model/disease_classifier.pt",
        labelmap_path: str = "jihye/model/id2label.pkl"
    ):
        # 1) Embedding, VectorDB, LLM 초기화
        self.emb = Embedding()
        self.db  = db
        self.llm = LLM()

        # # 2) 분류기 LabelMap 로드
        # with open(labelmap_path, "rb") as f:
        #     self.id2label = pickle.load(f)
        # num_labels = len(self.id2label)

        # # 3) BERT 분류기 로드
        # self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        # self.clf       = AutoModelForSequenceClassification.from_pretrained(
        #     "klue/bert-base", num_labels=num_labels
        # ).to(DEVICE)
        # self.clf.load_state_dict(torch.load(model_path, map_location=DEVICE))
        # self.clf.eval()

    def classify(self, question: str, top_k: int = 3):

        enc = self.tokenizer(
            question,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.clf(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE)
            )
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        top_ids = probs.argsort()[-top_k:][::-1]
        return [(self.id2label[i], float(probs[i])) for i in top_ids]

    def _build_prompt(self, question: str, disease: str, snippets: list[str]) -> str:

        prompt = """너는 질병을 특성을 알려주는 질병 에이전트야. 사용자의 질문을 보고 예측된 질병명과 관련 정보를 바탕으로
        해당 질병의 정의와 증상 그리고 원인에 대해 사용자에게 말하듯이 설명해줘."""
        
        prompt = f"사용자 질문: {question}\n"
        prompt += f"질병명: {disease}\n"
        if snippets:
            prompt += "관련 정보:\n"
            for s in snippets:
                prompt += f"- {s}\n"
        prompt += "위 내용을 바탕으로, 해당 질병의 정의와 증상 그리고 원인만 간단히 설명해야돼."
        return prompt

    def answer(self, diseases, question: str, top_k: int = 3) -> str:

        # 1) 질병 분류
        # preds = self.classify(question, top_k=top_k)
        # diseases = [name for name, _ in preds]

        # 2) 질병별 RAG 검색
        snippets_per_disease = {}
        for d in diseases:
            # 질병명과 질문을 결합해 입력 임베딩
            emb_input = f"{question} {d}"
            # q_emb = np.array([self.emb.embed(emb_input)]) # , dtype="float32")
            # idxs, _ = self.db.search(emb_input, search_type="similarity", top_k=2)
            docs = self.db.similarity_search(emb_input, k=2)

            # snippets_per_disease[d] = [self.db.texts[i] for i in idxs[0]]
            
            snippets_per_disease[d] = [doc.page_content for doc in docs]        

        # 3) 각 질병별로 LLM 프롬프트 생성 및 호출
        responses = []
        for d in diseases:
            prompt = self._build_prompt(question, d, snippets_per_disease.get(d, []))
            resp = self.llm.generate(prompt)
            responses.append(f"{d}:\n{resp}")

        # 4) 결과 병합
        return "\n\n".join(responses)
