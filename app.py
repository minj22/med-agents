import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr
import json
from langchain_community.vectorstores import FAISS
from langchain_upstage import ChatUpstage, UpstageEmbeddings

from dotenv import load_dotenv
import os
import re
from pre_agent.stage2 import PrescriptionAgent
from treat_agent.project2.agent.jinryo_agent import JinryoAgent
from treat_agent.project2.rag.retriever import retrieve_rag_data
from disease_agent.agent.disease_classifier import DiseaseAgent
from disease_agent.llm import LLM
from ex_agent.ex_agent import create_agent

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

api_key_1 = os.getenv("UPSTAGE_KEY_1")
api_key_2 = os.getenv("UPSTAGE_KEY_2")
api_key_3 = os.getenv("UPSTAGE_KEY_3")
api_key_4 = os.getenv("UPSTAGE_KEY_4")
embedding_key = os.getenv("EMBEDDING_KEY")

embedding_model = UpstageEmbeddings(api_key=embedding_key, model="embedding-query")

# explain_db = FAISS.load_local("db/explain_db", embedding_model, allow_dangerous_deserialization=True)
# prescript_db = FAISS.load_local("db/prescript_db", embedding_model, allow_dangerous_deserialization=True)
disease_db = FAISS.load_local("db/treat_db", embedding_model, allow_dangerous_deserialization=True)
# disease_db = FAISS.load_local("db/disease_db", embedding_model, allow_dangerous_deserialization=True)

# # retriever 정의
# explain_retriever = explain_db.as_retriever()
# prescript_retriever = prescript_db.as_retriever()
# treat_retriever = treat_db.as_retriever()
# disease_retriever = disease_db.as_retriever()


# 모델 경로
MODEL_PATH = "saved_model/bert_disease_model"

# 1. 모델 및 토크나이저 로딩
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


# 2. 클래스 ID → 이름 매핑 로드
# with open("./bert_disease_model/id2label.json", "r", encoding="utf-8") as f:
#     id2label = json.load(f)
id2label = model.config.id2label  

# 3. 예측 함수 정의
def predict_disease(question, history):
    print(question)
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    top3 = torch.topk(probs, k=3)
    top3_ids = top3.indices.tolist()
    top3_probs = top3.values.tolist()

    result_lines = []
    diease_list = []
    for i, (idx, prob) in enumerate(zip(top3_ids, top3_probs), start=1):
        label = id2label[idx] if isinstance(id2label, dict) else id2label[str(idx)]
        diease_list.append(label)
        result_lines.append(f"**{i}. {label}** — {prob*100:.2f}% 확률")

    top1_diease = diease_list[0]

    print(top1_diease)
    # stage2
    # kangjin agent (설명)
    
    ex_agent = create_agent()
    ex_answer = ex_agent.chat(top1_diease)
    
    # ex_answer = ex_answer['content']
    print(ex_answer)
    print("ex 완료")
    
    
    # minji agent (처방)
    prescipt_agent = PrescriptionAgent(
        json_dir="data/1.질문",
        embedding_api_key=embedding_key,
        chat_api_key=api_key_1
    )
    
    prescipt_agent.build_vectorstore(persist_dir='db/pre_db')
    
    top_docs = prescipt_agent.search_similar_docs(top1_diease)
    prompt = prescipt_agent.build_prompt(top1_diease, diease_list, top_docs)

    prescipt_answer = prescipt_agent.generate_response(prompt)

    print(prescipt_answer)
    print("pre 완료")

    # chaemin agent (진료)

    treat_agent = JinryoAgent(api_key=api_key_1)
    
    rag_qas_per_disease = []
    for disease in diease_list:
        qas = retrieve_rag_data("chaemin/project2/rag_embeddings.pkl", disease, question, top_k=3)
        rag_qas_per_disease.append(qas)
    
    print("treat search 완료")
            
    prompt = treat_agent.build_prompt(question, diease_list, rag_qas_per_disease)
    
    treat_answer = treat_agent.generate_response(prompt)
    
    print(treat_answer)
    print("tre 완료")

    
    # jihye agent (질병)
    dis_agent = DiseaseAgent(disease_db)
    dis_answer = dis_agent.answer(diseases= diease_list, question=question, top_k=3)
    
    print(dis_answer)
    print("dis 완료")
    
    # ex_answer = str(ex_answer).split("\n")[-1].split("id")[0]
    ex_answer = re.sub(r"<think>.*?</think>", "", str(ex_answer), flags=re.DOTALL).strip()
    # ex_answer = re.search(r'content="(.*?)"\s+id=', ex_answer, re.DOTALL)
    if 'content="' in ex_answer:
        ex_answer = ex_answer.split('content="')[1]
    
    ex_answer = ex_answer.split('id')[0]
    
    
    ex_answer = ex_answer.replace("\n\n", "")
    ex_answer = ex_answer.replace("content='", "")
    ex_answer = ex_answer.replace("\n\n", "")
    
    print(ex_answer)
    treat_answer = str(treat_answer).split("</think>")[1]
    
    final_result =  "".join([ex_answer, str(prescipt_answer), treat_answer, str(dis_answer)])
    print(final_result)
    history = history + [
        {"role": "user", "content": question},
        # {"role": "assistant", "content": "\n".join(result_lines)} # stage 1 result
        {"role": "assistant", "content": final_result}  # stage2 result
    ]
    return "", history


# 4. Gradio UI 구성

with gr.Blocks() as demo:
    gr.Markdown("## 질병 상담 Agent\n질병 관련 궁금한 점을 입력해 주세요.")

    with gr.Row():
        chatbot = gr.Chatbot(label="질병 상담 내역", type="messages", height=600)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="예: 고막염은 왜 생기나요?",
            show_label=False,
            scale=8,
            lines=2,
            container=True,
            autofocus=True
        )
        submit = gr.Button("질문하기", variant="primary", scale=2)


    msg.submit(fn=predict_disease, inputs=[msg, chatbot], outputs=[msg, chatbot])
    submit.click(predict_disease, inputs=[msg, chatbot], outputs=[msg, chatbot])


if __name__ == "__main__":
    demo.launch()