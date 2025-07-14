import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_core.messages import HumanMessage


class PrescriptionAgent:
    def __init__(self, json_dir, embedding_api_key, chat_api_key):
        self.json_dir = json_dir
        self.embedding_model = UpstageEmbeddings(api_key=embedding_api_key, model="embedding-query")
        self.chat_model = ChatUpstage(api_key=chat_api_key, model="solar-pro2-preview", reasoning_effort="high")
        self.docs = []
        self.chunks = []
        self.vectorstore = None

    def load_json_docs(self):
        all_docs = []
        for root, _, files in os.walk(self.json_dir):
            for file in files:
                if file.endswith(".json"):
                    try:
                        with open(os.path.join(root, file), encoding="utf-8") as f:
                            data = json.load(f)
                            disease = data["disease_name"]["kor"]
                            category = data.get("disease_category", "")
                            question = data.get("question", "").strip()
                            answer = data.get("answer", "").strip()
                            if disease and question and answer:
                                all_docs.append({
                                    "disease": disease,
                                    "category": category,
                                    "question": question,
                                    "answer": answer
                                })
                    except Exception as e:
                        print(f"❗ Error in {file}: {e}")
        print(f"유효 문서 수: {len(all_docs)}")
        self.docs = all_docs

    def filter_docs_by_disease(self, predicted_diseases):
        return [d for d in self.docs if d["disease"] in predicted_diseases]

    def chunk_documents(self, filtered_docs, chunk_size=300, chunk_overlap=30):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = []
        for d in filtered_docs:
            content = f"질병명: {d['disease']}\n카테고리: {d['category']}\n질문: {d['question']}\n답변: {d['answer']}"
            doc = Document(page_content=content, metadata={"disease": d["disease"]})
            self.chunks.extend(splitter.split_documents([doc]))
        print(f"생성된 Chunk 수: {len(self.chunks)}")

    def build_vectorstore(self, persist_dir="./chroma_temp"):
        self.vectorstore = FAISS.load_local("db/treat_db", self.embedding_model, allow_dangerous_deserialization=True)
        # FAISS.from_documents(self.chunks, self.embedding_model, persist_directory=persist_dir)
        # print("벡터 DB 구축 완료")

    def search_similar_docs(self, query, k=3):
        top_docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in top_docs]

    def build_prompt(self, symptom, predicted_diseases, top_contents):
        return f"""
        <|im_start|>system
        너는 환자의 증상과 예측된 질병을 바탕으로 **직접 처방을 제공하는 에이전트**야.

        너의 역할은 다음과 같아:

        - 환자의 증상을 보고 가장 가능성 높은 질병을 추정해.
        - 해당 질병의 원인과 특징을 간단히 설명해.
        - 환자가 스스로 해볼 수 있는 **자가 관리 방법, 초기 처방**을 구체적으로 안내해.
        - 병원에 가야 하는 기준이 있다면 반드시 알려줘.
        - 설명은 너무 길거나 딱딱하지 않게, **직접 상담하듯 부드럽고 친절하게** 말해줘.
        - 전문 용어는 최대한 피하고, 실생활 표현을 써.
        - `<think>`나 내부 설명은 절대 출력하지 마.
        - 참고 문서의 내용은 꼭 반영하고, 믿을 수 있는 정보를 제공해.

        응답은 하나의 자연스러운 문장 흐름으로 작성하고, 말투는 다음과 같은 식으로 써야 해:
        - "~해보시는 게 좋아요"
        - "~하는 것도 도움이 됩니다"
        - "~라면 병원에 방문하시는 걸 권장드려요"
        <|im_end|>

        <|im_start|>user
        증상:
        {symptom}

        예측된 질병:
        {', '.join(predicted_diseases)}

        참고 문서:
        {chr(10).join(top_contents)}

        이 정보를 바탕으로 환자에게 바로 말하듯 처방을 알려줘.
        <|im_end|>
        <|im_start|>assistant
        """.strip()

    def generate_response(self, prompt):
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        content = response.content
        if "</think>" in content:
            content = content.split("</think>", 1)[-1].strip()
        return content
        
        
if __name__ == "__main__":
    agent = PrescriptionAgent(
        json_dir="answered_output/1.질문",
        embedding_api_key="YOUR_API_KEY",
        chat_api_key="YOUR_API_KEY"
    )

    query = "무릎이 아프고 계단 오를 때 힘들어요"
    predicted_diseases = ['관절염', '류마티스 관절염', '골관절염']

    agent.load_json_docs()
    filtered = agent.filter_docs_by_disease(predicted_diseases)
    agent.chunk_documents(filtered)
    agent.build_vectorstore()

    top_docs = agent.search_similar_docs(query)
    prompt = agent.build_prompt(query, predicted_diseases, top_docs)
    print("프롬프트:\n", prompt)

    answer = agent.generate_response(prompt)
    print("\n 처방 응답:\n", answer)
