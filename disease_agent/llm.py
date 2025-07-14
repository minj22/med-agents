from langchain_upstage import ChatUpstage
import os
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self, model="solar-pro2-preview", do_healthcheck=True):
        self.llm = ChatUpstage(api_key=os.getenv("UPSTAGE_KEY_1"), model=model)
        if do_healthcheck:
            self._health_check()

    def _health_check(self):
        try:
            response = self.llm.invoke("hi")
            print("[LLM HealthCheck] 성공:", response.content[:30], "...")
        except Exception as e:
            raise RuntimeError(f"[LLM HealthCheck] 실패: {e}")

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

