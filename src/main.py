import torch
import tiktoken
from src.model import (GPTModel, create_dataloader_v1, train_model_simple, generate, text_to_token_ids, token_ids_to_text, GPTDatasetV1)


GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": 0.1, 
    "qkv_bias": False, 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG_124M)
model.to(device)

checkpoint = torch.load("model_and_optimizer.pth", map_location = device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

