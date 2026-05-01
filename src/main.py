"""
Main Interface for development right now. 

To use, python -m src.main --mode run
"""

import torch
import tiktoken
import argparse
from src.model import GPTModel
from src.data import GPTDatasetV1, create_dataloader_v1
from src.train import train_model_simple
from src.generation import generate, text_to_token_ids, token_ids_to_text

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

def load_model(device):

    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.to(device)

    checkpoint = torch.load("model_and_optimizer.pth", map_location = device)
    model.load_state_dict(checkpoint["model_state_dict"])


    model.eval()
    return model


def run_model(device):
    model = load_model(device)
    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = input("\n Enter prompt: ")
    while prompt != "q":
        token_ids = text_to_token_ids(prompt, tokenizer).to(device)

        output = generate(
            model = model,
            idx = token_ids,
            max_new_tokens = 100,
            context_size = GPT_CONFIG_124M["context_length"]
        )
        print(token_ids_to_text(output, tokenizer))
        prompt = input("\n Enter prompt: ")

    print("Ended")
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description="Train or Run")
    parser.add_argument("--mode", choices = ["train", "run"], required = True, help = "Train or run")
    args = parser.parse_args()

    if args.mode == "train":
        pass
        #train(device)
    elif args.mode == "run":
        run_model(device)


if __name__ == "__main__":
    main()
    

