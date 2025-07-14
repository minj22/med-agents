import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain.tools.retriever import create_retriever_tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Sequence, TypedDict
from pydantic import BaseModel, Field #v1 기반이래


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class GradeModel(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


class RAGChatbot:
    def __init__(self, vector_store_path="db/pre_db"):
        load_dotenv()
        
        # API 키
        self.api_key = os.getenv("UPSTAGE_KEY_1")
        embedding_key = os.getenv("EMBEDDING_KEY")
        
        # 모델 설정
        self.llm = ChatUpstage(api_key=self.api_key,
                               model="solar-pro2-preview",
                               reasoning_effort="high",
                            )
        embedding_model = UpstageEmbeddings(api_key=embedding_key, model="embedding-query")
        
        # 검색 도구
        retriever = FAISS.load_local(vector_store_path,
                                     embedding_model,
                                     allow_dangerous_deserialization=True
                                    ).as_retriever()
        self.retriever_tool = create_retriever_tool(retriever, "retrieve_diease", "Search information about diseases")
        
        # 그래프 생성
        self.graph = self._build_graph()
    
    def _build_graph(self):
        flow = StateGraph(AgentState)
        
        flow.add_node("agent", self._agent)
        flow.add_node("retrieve", ToolNode([self.retriever_tool]))
        flow.add_node("rewrite", self._rewrite)
        flow.add_node("generate", self._generate)
        
        flow.add_edge(START, "agent")
        flow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
        flow.add_conditional_edges("retrieve", self._grade_documents, {"generate": "generate", "rewrite": "rewrite"})
        flow.add_edge("rewrite", "agent")
        flow.add_edge("generate", END)
        
        return flow.compile()
    
    def _agent(self, state):
        tools = [convert_to_openai_tool(self.retriever_tool)]

        # model = self.llm.bind_tools([self.retriever_tool])
        model = self.llm.bind(
                tools=tools,
                tool_choice="auto",  # 자동 툴 선택
                # parallel_tool_calls=False도 가능
        )
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    
    def _grade_documents(self, state):
        # llm_with_tool = self.llm.with_structured_output(GradeModel)
        
        
        parser = PydanticOutputParser(pydantic_object=GradeModel)
        
        # prompt = PromptTemplate(
        #     template="Document: {context}\nQuestion: {question}\nIs this document relevant? Answer 'yes' or 'no'.",
        #     input_variables=["context", "question"]
        #             # partial_variables={"format_instructions": parser.get_format_instructions()}, 
        # )
        prompt = PromptTemplate(
            template=(
                    "{format_instructions}\n\n"
                    "Document:\n{context}\n\n"
                    "Question:\n{question}\n\n"
                    "Answer the question strictly in the JSON format shown above."
                ),
            input_variables=["context", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        chain = prompt | self.llm | parser

        result = chain.invoke({"question": question, "context": docs})
        # result = (prompt | llm_with_tool).invoke({"question": question, "context": docs})
        return "generate" if result.binary_score == "yes" else "rewrite"
    
    def _rewrite(self, state):
        question = state["messages"][0].content
        msg = HumanMessage(content=f"Rewrite this question better: {question}")
        response = self.llm.invoke([msg])
        return {"messages": [response]}
    
    def _generate(self, state):
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        
        prompt = ChatPromptTemplate([("human", 
            "Answer the question using this context. Answer in Korean.\n"
            "Question: {question}\nContext: {context}\nAnswer:")])
        
        response = (prompt | self.llm | StrOutputParser()).invoke({"question": question, "context": docs})
        return {"messages": [response]}
    
    def chat(self, message):
        """간단한 채팅 인터페이스"""
        inputs = {"messages": [("user", message)]}
        result = self.graph.invoke(inputs)
        return result["messages"][-1].content if isinstance(result["messages"][-1], str) else result["messages"][-1]


# 간단한 사용법
def create_agent(vector_store_path="db/pre_db"):
    return RAGChatbot(vector_store_path)


if __name__ == "__main__":
    chatbot = create_agent()
    response = chatbot.chat("안녕하세요!")
    print(response)