import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

from config import *
from data.loader import load_records, prepare_data
from model.classifier import TextDataset, build_model, save_model

def train():
    torch.manual_seed(SEED)

    recs = load_records(JSON_DIR)
    train, test, label2id, id2label = prepare_data(recs, SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(TextDataset(train, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(MODEL_NAME, num_labels=len(label2id)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 완료 - 평균 Loss: {avg_loss:.4f}")

    save_model(model, MODEL_PATH)
    print("모델 저장 완료")

    import pickle
    with open("label2id.pkl", "wb") as f:
        pickle.dump(label2id, f)
    print("label2id.pkl 저장 완료")

if __name__ == "__main__":
    train()