
import sys
import os
from models.llama3_1.reference_impl.model import *

#sys.path.append(os.path.join(os.getcwd()))
def show_paths():
    print("cur work path:", os.getcwd())
    print("sys path", sys.path)

def test_rope():
    head_dim=8
    seq_len=10
    theta=10000
    freq_cis = precompute_freqs_cis(head_dim, seq_len)
    print(freq_cis)

if __name__ == "__main__":
    show_paths()
    test_rope()