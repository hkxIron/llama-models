from typing import Callable

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

if __name__ == "__main__":
    show_paths()
    setup_seed(3407)
    test_rope()
    test_sample_top_p()