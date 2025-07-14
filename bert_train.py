import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report

import warnings
warnings.filterwarnings("ignore")
from transformers import TrainerCallback
import matplotlib.pyplot as plt

class MetricLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.eval_f1 = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_loss.append((state.epoch, logs['loss']))
            if 'eval_loss' in logs:
                self.eval_loss.append((state.epoch, logs['eval_loss']))
            if 'eval_f1' in logs:
                self.eval_f1.append((state.epoch, logs['eval_f1']))

def load_nested_jsons(base_path="data/1.질문"):
    data = []
    for disease_category in os.listdir(base_path):
        cat_path = os.path.join(base_path, disease_category)
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
                            with open(file_path, 'r', encoding='utf-8') as f:
                                j = json.load(f)
                                question = j.get("question", "").strip()
                                disease_kor = j.get("disease_name", {}).get("kor", "").strip()
                                if question and disease_kor:
                                    data.append({"question": question, "disease_name": disease_kor})
                        except Exception as e:
                            print(f"오류 in {file_path}: {e}")
    return pd.DataFrame(data)

# 1. 데이터 로딩
df = load_nested_jsons("data/1.질문")
print(f"총 {len(df)}개의 샘플 로딩 완료")

# 2. 라벨 매핑 (전체 질병 사용)
all_labels = sorted(df['disease_name'].unique())
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['disease_name'].map(label2id)
# print(df['label'].value_counts())
# print(df['disease_name'].value_counts())

# 3. Train-Test Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # stratify=df['label'],

# 4. Dataset 변환
train_dataset = Dataset.from_pandas(train_df[['question', 'label']])
test_dataset = Dataset.from_pandas(test_df[['question', 'label']])

# 5. Tokenizer 및 Tokenization
model_name = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples['question'], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)


# 1. 모델 정의
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 2. 학습 설정
training_args = TrainingArguments(
    output_dir="saved_model/train",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# 3. 평가 함수
def compute_metrics(pred):
    preds = torch.argmax(torch.tensor(pred.predictions), dim=-1)
    labels = torch.tensor(pred.label_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

logger_callback = MetricLoggerCallback()

# 4. Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    # optimizers=(AdamW(model.parameters(), lr=5e-5), None)
    callbacks=[logger_callback]

)


# 5. 학습 실행
trainer.train()

# 6. 모델 저장
trainer.save_model("saved_model/bert_disease_model")
tokenizer.save_pretrained("saved_model/bert_disease_model")

# 7. 평가 결과 출력
eval_results = trainer.evaluate()
print("Macro 평균 성능 지표:", eval_results)

# 8. 상세 리포트
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)
y_true = test_df['label'].tolist()

# print("\n=== 분류 상세 리포트 ===")
# print(classification_report(y_true, y_pred, target_names=all_labels))


# 시각화
epochs_loss = [ep for ep, _ in logger_callback.train_loss]
train_losses = [val for _, val in logger_callback.train_loss]
eval_losses = [val for _, val in logger_callback.eval_loss]
eval_f1s = [val for _, val in logger_callback.eval_f1]

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(epochs_loss, train_losses, label='Train Loss')
plt.plot(epochs_loss, eval_losses, label='Eval Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_loss, eval_f1s, label='Eval F1 Score', color='green')
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score Curve")
plt.legend()

plt.tight_layout()
plt.show()
