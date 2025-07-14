# API 키·엔드포인트·파라미터

import os
import torch
from dotenv import load_dotenv
load_dotenv()

# Upstage Solar Embedding API Key
SOLAR_API_KEY   = os.getenv("SOLAR_API_KEY")

# Upstage Solar Chat (Pro-2) API Key
SOLAR_CHAT_KEY  = os.getenv("SOLAR_CHAT_KEY")

# Device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
