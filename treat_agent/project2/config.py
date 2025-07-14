import os
from dotenv import load_dotenv
import torch

load_dotenv()  # .env 파일에서 환경변수 로드

MODEL_NAME = "klue/bert-base"
JSON_DIR = "./answered_output"
BATCH_SIZE = 16
EPOCHS = 1
LR = 2e-5
MAX_LEN = 128
SEED = 42
MODEL_PATH = "./saved_model.pt"
PICKLE_PATH = "./rag_embeddings.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY_1", "")