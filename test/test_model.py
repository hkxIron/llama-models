from models.llama3_1.reference_impl.model import * # 绝对导入
from util import setup_seed

def show_paths():
    import sys
    import os
    print("cur work path:", os.getcwd())
    print("sys path:", sys.path)
    #sys.path.append()

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

if __name__ == "__main__":
    show_paths()
    setup_seed(3407)
    test_rope()