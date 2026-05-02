import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def generate(model, idx, max_new_tokens, context_size, temperature = 0.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]

        #call the forward pass of GPTMODEL such that input enters model 
        with torch.no_grad():
            #call GPTModel.forward() which starts the GPT logic.
            logits = model(idx_cond) 
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        

        if temperature > 0.0:
            logits = logits/temperature

            logits = logits - logits.max(dim = -1, keepdim = True).values
            #the probability is just softmax of raw scores
            probabilities = torch.softmax(logits, dim = -1)

            idx_next = torch.multinomial(probabilities, num_samples = 1)            

        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)

        if idx_next == eos_id:
            break 
        idx = torch.cat((idx, idx_next), dim = 1)
    
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

