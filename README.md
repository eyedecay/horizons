Learning How LLMs Work

Part 1: Input data
Turn textx into tokens into token ids into token embeddings + positional embeddings = input embeddings

- Roughly how text preprocessing works (the manual regex way and using tiktoken)
- unknown token handling
- created a dataset used with DataLoader and created a dataloader

Part 2: Attention
Why self-attention? Other models needed to read full input and then "memorize" it. When translating text, the model can't look at full input and loses information. Self-attention can look at whole input by looking at the importance of different tokens. 

(The simple "not real" self-attention)
- Compute Attention Scores (Dot product between query and keys)
- Normalize using softmax function
- Context Vector: final result of self-attention for a token representing what information the token should carry

The real LLM attention mechanism

nn.Module is base Neural Network for Pytorch
super().__init__() for inheritance


Given embedding:
- every attention Layer has query, key, and value weights learned during training
- The embedding matrix is multipled by each weight matrix to get query (what a token looks for), key (what a token represents), value (what a token has)
- Attention score is the query and the key dot product (measuring similarity between a token and all other tokens), then normalized (scaled + softmax) to get weights. Weights then dot product with value to get final context vector. 

Causal Attention mask
- Used such that a token can only attend to iteslf and previous tokens, not future ones so during training, it can't "Cheat" 

Dropout Mask
- Masked out random positions to reduce overfitting
- Applied during training


Multi-Head Attention
- Each head has different weights, so they will learn different features
- Each head produces its own context vectors which are concatenated then applied a linear layer (out_proj) to mix them

Part 3: Architecture

- Layer Normalization: used to spped up training and stabilize by normalizing activations. For an input, layer computes mean and variance such that mean = 0 and variance = 1. 
    - Scale and Shift Weights in Layer Normalization are used to adjust the normalized output
- GELU: a non-linear activation function that sits in between linear layers. A FeedForward Module is coded into the transformer block that contains Linear Layer, GELU, and Linear Layer. 
- Activation functions add non-linearity to neural networks to learn complex patterns. 

nn.Sequential is a container that allows to build layers in neural nets
In a Linear layer, the input activations are used to learn weights using y = xW+b. The First Linear Layer expands the feature size, then GELU applies smooth filtering, then Second Layer compresses it and "summarizes" what is learned into the right dimensions. 

- Shortcut Connections: input is added directly to output to preserve original information and make gradients flow easier. 