from typing import Callable

import torch

from models.llama3_1.reference_impl.model import * # 绝对导入
from util import setup_seed
from models.llama3_1.reference_impl.generation import *

def my_decorator(func:Callable):
    def my_wrapper(*args, **kwargs):
        #start_time = time.time()
        print("="*30+f" {func.__name__} begin "+"="*30)
        result = func(*args, **kwargs)
        print(f"{func.__name__} end\n\n")
        #end_time = time.time()
        return result
    return my_wrapper

def show_paths():
    import sys
    import os
    print("cur work path:", os.getcwd())
    print("sys path:", sys.path)
    #sys.path.append()

@my_decorator
def test_rope():
    head_dim=6
    seq_len=4
    theta=10000
    freq_cis = precompute_freqs_cis(head_dim, seq_len, theta)
    print("freq_cis size:",freq_cis.shape) # [seq_len, head_dim//2]
    print("freq_cis:",freq_cis)

    batch = 2
    head_num=3
    query = torch.randn([batch, seq_len, head_num, head_dim])
    print("query[0][0]:\n", query[0][0])
    print("query[0][1]:\n", query[0][1])
    print("query[1][1]:\n", query[1][1])
    query = query.detach().clone()
    query_rope, key_rope = apply_rotary_emb(query, query, freqs_cis=freq_cis)
    #print("query_rope size:", query_rope.size())
    print("query_rope[0][0]:\n", query_rope[0][0])
    print("query_rope[0][1]:\n", query_rope[0][1])
    print("query_rope[1][1]:\n", query_rope[1][1])

@my_decorator
def test_sample_top_p():
    batch=2
    vocab=5
    probs = torch.randn((batch, vocab)).abs()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    print(probs)
    token_ids = sample_top_p(probs, p=0.9)
    print(token_ids)

@my_decorator
def test_transformer():
    vocab_size = 1000
    model_args = ModelArgs(
        dim=16, # 4096,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=2,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        rope_theta=10000,
        use_scaled_rope=False,
        max_batch_size=10,
        max_seq_len=128,
    )
    transformer_model = Transformer(model_args)
    print(transformer_model)
    """
    Transformer(
  (tok_embeddings): Embedding(1000, 16)
  (layers): ModuleList(
    (0-1): 2 x TransformerBlock(
      (attention): Attention(
        (wq): Linear(in_features=16, out_features=16, bias=False)
        (wk): Linear(in_features=16, out_features=8, bias=False)
        (wv): Linear(in_features=16, out_features=8, bias=False)
        (wo): Linear(in_features=16, out_features=16, bias=False)
      )
      (feed_forward): FeedForward(
        (w1): Linear(in_features=16, out_features=42, bias=False)
        (w2): Linear(in_features=42, out_features=16, bias=False)
        (w3): Linear(in_features=16, out_features=42, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (lm_head): Linear(in_features=16, out_features=1000, bias=False)
)
"""


    batch =2
    seq_len=10
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    print(tokens)
    start_pos = 0
    logits = transformer_model.forward(tokens, start_pos)
    print("logits shape:", logits.shape)
    print(logits)

@my_decorator
def test_llama():
    tokenizer = Tokenizer("../models/llama3_1/api/tokenizer.model")
    print("vocab size:", tokenizer.n_words) # 128256

    vocab_size = tokenizer.n_words
    max_seq_len = 50

    model_args = ModelArgs(
        dim=16, # 4096,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=vocab_size,
        multiple_of=2,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        rope_theta=10000,
        use_scaled_rope=False,
        max_batch_size=10,
        max_seq_len=max_seq_len,
    )
    transformer_model = Transformer(model_args)

    #batch =2
    #seq_len=10
    #token_ids = torch.randint(0, vocab_size, (batch, seq_len))
    #print(token_ids)
    #logits = transformer_model.forward(tokens, start_pos)

    llama = Llama(transformer_model, tokenizer, model_args)
    prompts = ["你是语音助手,北京在哪", "你好,你好"]

    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    max_gen_len = max_seq_len - max([len(x) for x in prompt_tokens])
    print("max_gen_len:", max_gen_len)
    temperature=0.01
    top_p=0.9
    logprobs=False

    token_ids = [] # List[List[]]
    token_logprobs = []
    decoded_tokens = []
    for result in llama.generate(model_input=ModelInput(batch_tokens=prompt_tokens), max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, logprobs=logprobs,):
        token_ids.append(result.token_id)
        if logprobs:
            decoded_tokens.append(result.text)
            token_logprobs.append(result.logprobs)

    print(f"gen token id len:{len(token_ids)}")
    import numpy as np
    token_ids = np.array(token_ids).T.tolist()
    batch_gen_tokens = []
    for index, tokens in enumerate(token_ids):
        text = tokenizer.decode(tokens)
        batch_gen_tokens.append(text) # 注意：这里有batch内最长与最短的生成的text处理
        print(f"index:{index} gen text:{text}")
    print("generate text:\n", batch_gen_tokens)

@my_decorator
def test_tokenizer():
    tokenizer = Tokenizer("../models/llama3_1/api/tokenizer.model")
    print(tokenizer.n_words)
    ids = tokenizer.encode("我是中国人", bos=True, eos=True)
    print(ids)
    text = tokenizer.decode(ids)
    print(text)

if __name__ == "__main__":
    show_paths()
    setup_seed(3407)

    test_llama()

    if False:
        test_rope()
        test_sample_top_p()

        test_tokenizer()
        test_transformer()
