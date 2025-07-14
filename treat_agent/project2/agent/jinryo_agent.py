from openai import OpenAI
from dotenv import load_dotenv
import os


class JinryoAgent:
    def __init__(self, api_key=os.getenv("UPSTAGE_KEY_1"), base_url="https://api.upstage.ai/v1", model="solar-pro2-preview"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def build_prompt(self, user_input, preds, rag_qas_per_disease):
        prompt = f"""
        너는 진료 에이전트야. 사용자의 증상을 듣고 다음 질병들을 예측했어: {', '.join(preds)}.
        아래 각 질병에 대해 관련된 Q-A 정보를 참고해서 최종 진단과 대응 방안을 3문장 이내로 알려줘.

        사용자 증상: {user_input}
        """
        for i, disease in enumerate(preds):
            prompt += f"\n### 질병 {i+1}: {disease}\n"
            qas = rag_qas_per_disease[i]
            for j, qa in enumerate(qas):
                prompt += f"\nQ{j+1}: {qa['question']}\nA{j+1}: {qa['answer']}\n"

        prompt += "\n위 정보를 종합해 가장 가능성 높은 진단과 대응 방안을 알려줘."
        return prompt

    def generate_response(self, prompt):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
            stream=True
        )
        result = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
        return result.strip()