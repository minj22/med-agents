from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document
import os, json
from dotenv import load_dotenv
import time

# .env 파일 로드
load_dotenv()

# 환경 변수 가져오기
api_key = os.getenv("EMBEDDING_KEY")

def load_documents_from_folder(folder_path: str):
    explain_docs = []
    prescript_docs = []
    treat_docs = []
    diease_docs = []
    
    
    for disease_category in os.listdir(folder_path):
        cat_path = os.path.join(folder_path, disease_category)
        if not os.path.isdir(cat_path):
            continue
        for disease_name in os.listdir(cat_path):
            dis_path = os.path.join(cat_path, disease_name)
            if not os.path.isdir(dis_path):
                continue
            for topic in os.listdir(dis_path):  # 예방, 원인, 증상 등
                topic_path = os.path.join(dis_path, topic)
                if not os.path.isdir(topic_path):
                    continue
                for file in os.listdir(topic_path):
                    if file.endswith(".json"):
                        file_path = os.path.join(topic_path, file)
                        try:
                            with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    intetion = data["intention"]
                                    content = f"[질문: {data['question']}]\n[답변: {data['answer']}]"
                                    metadata = {
                                        "disease_category": data["disease_category"],
                                        "disease": data["disease_name"]["kor"],
                                        "intention": intetion,
                                        "gender": data["participantsInfo"]["gender"],
                                        "age": data["participantsInfo"]["age"],
                                        "occupation" : data["participantsInfo"]["occupation"]
                                    }
                                    
                                    if intetion == '식이/생활' or intetion == '원인' or intetion =='진단':
                                        explain_docs.append(Document(page_content=content, metadata=metadata))
                                        print("OKAY1")
                                    elif intetion == '증상' or intetion == '검진':
                                        prescript_docs.append(Document(page_content=content, metadata=metadata))
                                        print("OKAY2")
                                        
                                    elif intetion == '약물' or intetion == '예방' or intetion == '운동' or intetion == '재활' or intetion == '치료진':
                                        treat_docs.append(Document(page_content=content, metadata=metadata))
                                        print("OKAY3")
                                        
                                    else:
                                        diease_docs.append(Document(page_content=content, metadata=metadata))
                                        print(metadata['disease'])
                                        print("OKAY4")
                                        
                        except Exception as e:
                            print(f"오류 in {file_path}: {e}")
                            
    return explain_docs, prescript_docs, treat_docs, diease_docs
    

if __name__ == "__main__":
    embedding_model = UpstageEmbeddings(api_key=api_key, model="embedding-query")
    folder_path = "data/1.질문"  # JSON 폴더 경로

    explain_docs, prescript_docs, treat_docs, diease_docs = load_documents_from_folder(folder_path)

    print("load 완료")
    # 벡터 DB 생성 및 저장
    
    ex_db = FAISS.from_documents(
        documents=explain_docs,
        embedding=embedding_model,
    )
    ex_db.save_local("db/ex_db")

    print("ex db 완료")
    
    # pre_db = FAISS.from_documents(
    #     documents=prescript_docs,
    #     embedding=embedding_model,
    #     persist_directory="db/prescript_db"  # 저장 경로
    # )

    # # chroma_pre_db.persist()   

    # print("pre db 완료")
 
    # chroma_treat_db = FAISS.from_documents(
    #     documents=treat_docs,
    #     embedding=embedding_model,
    #     persist_directory="db/treat_db"  # 저장 경로
    # )

    # # chroma_treat_db.persist()
    
    # print("treat db 완료")

    # chroma_dis_db = FAISS.from_documents(
    #     documents=diease_docs,
    #     embedding=embedding_model,
    #     persist_directory="db/diease_db"  # 저장 경로
    # )

    # # chroma_dis_db.persist()
    
    print("벡터 DB 생성 및 저장 완료")