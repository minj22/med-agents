# 🏥 실제 진료 흐름을 모사한 LLM 기반 Multi-Agent

**사용자의 질문 및 증상에 맞는 전문 Agent의 응답 처리**


## 🎥 시뮬레이션 동영상

![medAgent_simulation](https://github.com/user-attachments/assets/6ba88085-95b6-4844-b6ab-e665f8748a44)

---


## 📌 프로젝트 개요
본 프로젝트는 LLM 기반 Multi-Agent 시스템을 구축하여 실제 진료 흐름을 재현하고 사용자의 증상에 따라 질병을 자동 분류하고 맞춤형 의료 정보를 제공하는 것을 목표로 합니다.  
단일 LLM 또는 RAG 시스템으로는 설명,치료,처방,진단 등 다양한 의료 목적을 동시에 충족시키기 어려운 한계를 해결하기 위해 기능별 Agent를 Multi-Agent 구조로 분리하여 설계했습니다.

---


## 💡 목표
- **단계별 맞춤형 의료 상담 자동화**
- **증상 기반 질병명 자동 분류**
- **동일 RAG DB 기반의 질의응답 생성**
- **의료 정보 접근성 향상**

---


## ✔️ 차별점
- 단일 LLM이 아닌 역할 분리형 Multi-Agent 구조 설계
- Agentic RAG를 활용해 LLM의 의사결정 능력과 도구 활용 결합
- 한국어 의료 데이터 최적화를 위해 SOLAR 임베딩 모델 사용

---


## ✔️ 데이터셋
- **출처**: AI HUB 의료 Q&A 데이터셋
- **규모**: 5.51GB
- **구성**: 질환 → 질병 → 응답 유형(진단, 치료, 증상)
- **문제점**:
  - 질의와 응답 데이터 매칭 불완전
  - 생소하거나 희귀 질병 비중 높음
  - 대용량 텍스트로 인한 전처리 필요

---


## ✔️ 데이터 전처리 및 증강
1. **필터링**  
   - 모델 학습에 필요한 핵심 필드(question, disease...) 추출
   - 희귀하거나 데이터 수가 적은 질병 제거 
2. **증강**  
   - Qwen3-30B-a3b 모델 기반 의료 지식을 활용해 응답 데이터 생성
3. **벡터화 및 Chunking**  
   - 300자 이상 텍스트를 중복 30자 포함하여 분할
4. **라벨링**  
   - 질병명 라벨 부여 후 RAG DB 구축

---


## 🔧 주요 기술 스택
- **BERT**: 한국어 의료 데이터 질병 분류기
- **SOLAR-PRO2**: 한국어 최적화 임베딩 모델
- **RAG**: 외부 지식 검색 + LLM 응답 생성
- **Agentic RAG**: LangGraph 기반 LLM의 도구 사용,판단,응답 전략

---


## ✔️ 모델 아키텍처
<img width="583" height="560" alt="image" src="https://github.com/user-attachments/assets/51309d39-4396-467b-bff1-29c3ebb859e3" />

### Stage 1: 질병 예측
- BERT 기반 분류기로 사용자 질의에서 **확률 상위 3개 질병** 예측

### Stage 2: Multi-Agent 질의응답
- **질병 Agent**: 질병 정보 제공
- **처방 Agent**: 치료 방법 및 약물 정보 안내
- **진단 Agent**: 진단 방법 제공
- **설명 Agent**: 질병 설명 및 부가 정보 제공
- **최종 Agent**: 각 Agent 응답 종합

---


## 💡 기대효과
- 의료 정보 접근성 향상
- 질병 예측 → 정보 검색 → 진단 응답 생성까지 일관된 흐름 제공
- 역할별 Agent로 응답 정확도 향상

---


## ⚠️ 한계
- 데이터셋에 없는 생소한 질병에 취약
- 사용자의 증상 표현이 모호할 경우 분류 정확도 저하
- 고정 질병 목록 기반으로만 인식 가능 → 일반화 한계

---

## 🔗 팀원 소개

<table>
  <tr align="center">
    <td><img src="https://github.com/lkj626.png" width="220"/></td>
    <td><img src="https://github.com/.png" width="220"/></td>
    <td><img src="https://github.com/minj22.png" width="220"/></td>
    <td><img src="https://github.com/limjihyee.png" width="220"/></td>
    <td><img src="https://github.com/Chaemin78.png" width="220"/></td>
  </tr>
  <tr align="center">
    <td><a href="https://github.com/lkj626">이강진</a></td>
    <td><a href="https://github.com/">정환길</a></td>
    <td><a href="https://github.com/minj22">양민지</a></td>
    <td><a href="https://github.com/limjihyee">임지혜</a></td>
    <td><a href="https://github.com/Chaemin78">임채민</a></td>
  </tr>
  <tr align="center">
    <td>최종 에이전트 설계</td>
    <td>PM 및 데이터 증강</td>
    <td>설명 에이전트 설계</td>
    <td>질병 에이전트 설계</td>
    <td>진단 에이전트 설계</td>
  </tr>
</table>

---

## ⚙️ 환경 세팅 & 실행 방법

### 1. 저장소 클론
```bash
git clone https://github.com/minj22/med-agents.git
cd med-agents
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. RAG DB 구축
```bash
python build_db.py
```

### 4. 서비스 실행
```bash
python app.py
```
