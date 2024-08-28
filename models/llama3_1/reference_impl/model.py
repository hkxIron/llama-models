# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Optional, Tuple

#import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding,
# )
from torch import nn
from torch.nn import Embedding
from torch.nn import Linear

#print(__name__)
from ..api.args import ModelArgs

# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        # 波长
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / ( high_freq_factor - low_freq_factor )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


"""
precompute_freqs_cis中的cis是 "cosine" 和 "sine" 的缩写，它经常在数学中使用来表示复数的极坐标形式。
具体来说，给定一个角度theta，其对应的复数可以表示为：
cis(theta) = cos(theta) + i*sin(theta), 即一般形式的欧拉公式
"cis" 表示的是一个复数，其实部是角度θ的余弦值，而虚部是角度θ的正弦值, theta表示幅角 ,这种表示方法在复数分析、信号处理等领域中非常有用。

因此，故名思义，该函数的目的是预计算一个复数频率张量。该函数有两个入参，dim和end。
dim就是每个attention_head中的维度，在这里就是head_dim = hidden/head_num=4096/32=128。
end是self.params.max_seq_len * 2，也就是4096，这也是Llama2最大的token处理数量。计算过程解释见注释：
"""
def precompute_freqs_cis(
    dim: int, # head_dim =128
    seq_length: int, # max_seq_len*2
    theta: float = 10000.0,
    use_scaled: bool = False
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        seq_length (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    # dim = head_dim = 128
    # seq_len = max_seq_len*2 = 4096
    # 幅角最小单位：10000^(-2i/dim)
    # torch.arange(0, dim, 2) [0, 2, 4, 6, 8, 10,..., 124, 126] 共64个
    # torch.arange(0, dim, 2)[: (dim // 2)] 保证是64个
    # freqs = [1/10000.0^(0/128), 1/10000.0^(2/128), 1/10000.0^(4/128), ..., 1/10000.0^(126/128)]
    # freqs.shape: [dim//2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # index_of_seq: [0, 1, 2, ..., 4095]
    # index_of_seq.shape:[seq_len]
    index_of_seq = torch.arange(seq_length, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)

    """
    freqs 得到 freqs和t的笛卡尔积
    freqs:[seq_length, embed_dim//2] =（4096，64）
    freqs = [[0, 0, 0,..., 0],
             [1/10000.0^(0/128), 1/10000.0^(2/128), 1/10000.0^(4/128), ..., 1/10000.0^(126/128)],
             [2/10000.0^(0/128), 2/10000.0^(2/128), 2/10000.0^(4/128), ..., 2/10000.0^(126/128)],
             ...,
             [4095/10000.0^(0/128), 4095/10000.0^(2/128), 4095/10000.0^(4/128), ..., 4095/10000.0^(126/128)]]
    其公式值为：
    [
        0*theta(0), 0*theta(1), ..., 0*theta(dim/2-1),
        1*theta(0), 1*theta(1), ..., 1*theta(dim/2-1),
        ...
        m*theta(0), m*theta(1), ..., m*theta(dim/2-1),
        ...
        seq_len*theta(0), seq_len*theta(1), ..., seq_len*theta(dim/2-1),
        ]
    """
    # index_of_seq.shape:[seq_len]
    # freqs:[dim//2]
    # freqs_angle:[seq_len, dim//2]
    freqs_angle = torch.outer(index_of_seq, freqs)

    """
    在PyTorch中，torch.polar用于通过极坐标（magnitude和angle）来创建一个复数张量。
    这个函数接受两个张量作为输入：一个张量包含复数的模（magnitude，也就是复数的长度），
    另一个张量包含复数的角度（angle，也就是复数的相位角），然后返回一个相应的复数张量。
    下面就是创建模长为1的，有不同相位角的复数张量。
    freqs_cis:[seq_len, dim//2]
    """
    freqs_cis = torch.polar(abs=torch.ones_like(freqs), angle=freqs_angle)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    """
    注意freqs_cis的维度并不是（4096，64），而是截取了seqlen的一部分，freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]。
    """
    # freqs_cis.shape = [1024, 64]
    # x.shape = [2, 1024, 32, 64]
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # freqs_cis:[seq_length, embed_dim//2]
    # x:[batch, query_seqlen, head_num, head_dim/2]
    # 将freqs_cis.shape变为[batch=1, query_seqlen=1024, head_num=1, head_dim/2=64]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


"""
与其它实现不同，meta的实现直接在复数空间相乘得到rope编码，即
f(q,m)=q_complex*e^(i*m*theta)
"""
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    # xq:[batch, seqlen, head_num, head_dim]
    # xk:[batch, seqlen, n_kv_head, head_dim]

    """
    将xq和xk的最后一个维度进行复数运算，得到新的xq和xk
    为了进行复数运算，需要将xq和xk的最后一个维度展开为2维
    例如，xq的形状为[2, seq_len, 32, 128], reshape后为[2, seq_len, 32 , 64, 2]
    view_as_complex函数可以将张量中的最后一维的两个元素作为实部和虚部合成一个复数

    xq:[batch, query_seqlen, head_num, head_dim]
    -> [batch, query_seqlen, head_num, head_dim/2, 2]
    torch.view_as_complex:其中输入张量的最后一个维度必须为2, 将相邻位置(q0,q1)作为复数的实部与虚部,其中偶数部分为实部，奇数部分为虚部
    具体而言，其中的复数为：[x0+j*x(1), x2+j*x3, ...., x(dim_2)+j*x(dim-1)], 长度为head_dim/2
    此处reshape会将相邻位置当成复数的实部与虚部
    xq_complex: [batch, query_seqlen, head_num, head_dim/2]
    """
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))

    # xk:[batch, seqlen, n_kv_head, head_dim]
    # -> [batch, key_seqlen, head_num, head_dim/2, 2]
    # xk_complex: [batch, key_seqlen, head_num, head_dim/2]
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    """
    freqs_cis:[seq_length, embed_dim//2]
    其值为：
    [
    0*theta(0), 0*theta(1), ..., 0*theta(dim/2-1),
    1*theta(0), 1*theta(1), ..., 1*theta(dim/2-1),
    ...
    m*theta(0), m*theta(1), ..., m*theta(dim/2-1),
    ...
    seq_len*theta(0), seq_len*theta(1), ..., seq_len*theta(dim/2-1)
    ]
    xq_complex: [batch, query_seqlen, head_num, head_dim/2]
    将freqs_cis.shape变为[batch=1, query_seqlen=1024, head_num=1, head_dim/2=64]
    """
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    """
    ROPE编码, f(q, m) = q_complex*e^(i*m*theta)
    其具有相对位置信息：<f(q,m), f(k,n)> = g(q,k,m-n) = (q.T)*R(n-m)*k
    即将xq转为复数后，与位置m的复数相乘，得到rope
    
    xq_complex: [batch, query_seqlen, head_num, head_dim/2]
    freqs_cis:  [batch=1, query_seqlen=1024, head_num=1, head_dim/2=64]
    view_as_real和view_as_complex相反，可以将张量中最后一维的复数拆出实部和虚部
    
    xq_complex*freqs_cis为复数相乘, 即模长相乘，幅角相加,由于freqs_cis模长为1,因此只有幅角相加
    xq_rope_complex.shape:[batch, seqlen, head_num, head_dim/2],结果为复数，有实部与虚部
    xq_rope_real.shape = [batch=2, seq_len, head_num=32 , head_dim/2=64, 2]
    xq_rope: flatten(3)将张量展平为[batch=2, seq_len, head_num=32, head_dim=128]，3代表从的第3个维度开始展平
    
    xq_rope:[batch, query_seqlen=1024, head_num, head_dim]
    xk_rope:[batch, query_seqlen=1024, head_num, head_dim]
    """
    xq_rope_complex = xq_complex * freqs_cis
    xq_rope_real = torch.view_as_real(xq_rope_complex)
    xq_rope = xq_rope_real.flatten(3) # 从第3维head_dim//2开始，将后面所有维（head_dim//2, 2）展平


    xk_rope = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    """
    最终,xq_rope的复数表示为 
    [batch==0, seq_len==0, head_num==0, head_dim= [ 
                                                   q0*cos(0*theta0)-q1*sin(0*theta0),
                                                   q1*cost(1*theta0)+q0*sin(1*theta0),
                                                   q2*cos(2*theta1)-q3*sin(2*theta1),
                                                   q3*cost(3*theta1)+q2*sin(3*theta1),
                                                   ...
                                                   q3*cost(m*theta1)+q2*sin(m*theta1),
                                                   ...
                                                   q(d-2)*cos(m*theta(dim/2))-q(d-1)*sin(m*theta(dim/2)),
                                                   q(d-1)*cos(m*theta(dim/2))-q(d-2)*sin(m*theta(dim/2))
                                                ]
  ]
    """
    return xq_rope.type_as(xq), xk_rope.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    torch.repeat_interleave(x, dim=2, repeats=n_rep)
    元素级别的复制，不是整体复制
    """
    bs, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """

        super().__init__()
        # n_kv_heads与n_heads默认相同
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        # 模型并行的大小
        #model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # GQA, MQA中，多个query共享一个kv,实现时为了方便将kv复制多份以进行attention
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # ColumnParallelLinear是一个在大规模并行训练中使用的术语，特别是在训练大型的深度学习模型，
        # 如Transformer模型时。在模型并行训练中，一个大型的矩阵（例如神经网络的权重矩阵）会被分割成不同的列，
        # 并分散到不同的计算设备（如GPU）上。
        #
        # 在ColumnParallelLinear的情况下，每个计算设备存储权重矩阵的一部分列，而不是整个矩阵。
        # 每个设备计算它自己的前向传播部分，并将结果发送给其他设备以进行进一步的处理或合并结果。
        # 对于反向传播和梯度计算，每个设备计算其自己列的梯度，并可能需要与其他设备交换信息以更新权重。
        #
        # 这种方式可以显著减少每个设备上的内存需求，并允许训练更大的模型，因为模型的不同部分可以分布在多个设备上。
        # ColumnParallelLinear和RowParallelLinear（另一种将权重矩阵按行划分的方法）是实现模型并行(张量并行)的两种常见策略。

        # self.wq = ColumnParallelLinear( in_features=args.dim,
        #     out_features=args.n_heads * self.head_dim, # 会在out_features维度并行
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wk = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim, # 会在out_features维度并行
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wv = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wo = RowParallelLinear(
        #     args.n_heads * self.head_dim, # 会在in_feature维度并行
        #     args.dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )

        self.wq = Linear( in_features=args.dim,
                                        out_features=args.n_heads * self.head_dim, # 会在out_features维度并行
                                        bias=False)
        self.wk = Linear(
            args.dim,
            self.n_kv_heads * self.head_dim, # 会在out_features维度并行
            bias=False,
            )
        self.wv = Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            )
        self.wo = Linear(
            args.n_heads * self.head_dim, # 会在in_feature维度并行
            args.dim,
            bias=False,
            )

        # kv_cache是缓存键值对，在训练过程中，我们只保存最近n个键值对
        # 按照最大的batch,最长的seq_len来分配cache内存
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads, # 注意是 local_kv_heads
                self.head_dim,
            )
        ) #.cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ) #.cuda()

    """
    大模型一般是分布式训练，这里涉及到几个概念。n_heads是注意力头的总个数，由于并行机制，每个进程会有n_local_heads个注意力头。
    由于计算当前位置的Attention Score依赖于之前所有的kv，因此需要将kv缓存下来。
    为了减少空间复杂度，可以对kv的头个数n_kv_heads进行调整，这个值一般小于等于n_heads，
    n_heads是n_kv_heads的整数倍，这个倍数也就是n_rep。
    相应的，每个进程会有n_local_kv_heads个注意力头。
    每个头的维度为head_dim=dim//n_heads。

    例如：n_heads=32，model_parallel_size（并行数量）= 4，n_kv_heads = 8，
    n_local_heads = 32/4， n_local_kv_heads = 8/4，n_rep = 32/8。
    """
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        batch_size, seq_len, hidden_size = x.shape # 在推理阶段，过了prompt prefilling阶段，每次输入的x长度为1, seq_len=1
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # xq:[batch, seq_len, head_num, head_dim]
        # xk:[batch, seq_len, n_kv_head, head_dim]
        # xv:[batch, seq_len, h_kv_head, head_dim]
        xq = xq.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim) # 注意：这里就是group query attention
        xv = xv.view(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        """
        对当前token的qkv计算位置编码ROPE
        注意：ROPE是乘性位置编码，在每一层attention时都需要显式加入位置信息,
        但在google原始encoder-decoder的Transfomer中，使用的是sin-cos-position,
        并且只在第一层加入位置信息,而rope是需要每层均加入位置信息
        
        xq:[batch, seq_len, head_num, head_dim]
        xk:[batch, seq_len, n_kv_head, head_dim]
        freqs_cis:[seq_length, embed_dim//2]
        """
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 缓存当前token的kv,将数据复制到xq的设备上
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        """
        训练阶段不能缓存kv cache，因为参数一直在变,此处因为meta给的是推理代码，所以可以用kv cache.
        start_pos也只是在推理阶段的参数，训练时无此参数
        """
        # 将当前的key/value缓存起来
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv

        # 取出前seqlen个token的kv缓存
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # repeat k/v heads if n_kv_heads < n_heads, 即group query attention或multi query attention
        # 将kv重复填充，使kv和q的头数个数相同
        """
        keys/value:[batch, cache_len+seq_len, n_local_kv_heads, head_dim] 
          => [batch, cache_len+seq_len, n_local_heads, head_dim] 
        """
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seq_len, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seq_len, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seq_len, head_dim)
        values = values.transpose( 1, 2 )  # (bs, n_local_heads, cache_len + seq_len, head_dim)
        """
        注意：在推理阶段，过了prompt prefilling后,采用kv cache,每次计算新输入的token的attention，seqlen均为1,因此xq的实际shape均为： (bs, n_local_heads, seq_len=1, head_dim),
        因此此时也不再需要mask,mask均为None, 这样大大节省了计算量
        """
        # scores:[batch, n_local_heads, seq_len, cache_len+seq_len]
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            # 推理阶段，只有在第一次prompt_prefill时，mask才不为None
            # 过了prompt_prefill阶段后，mask均为None, 每次只生成一个token
            # 如果mask存在，mask值为-inf
            scores = scores + mask  # (bs, n_local_heads, seq_len, cache_len + seq_len)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 注意：在llama inference中，attention里并没有dropout
        # scores:[batch, n_local_heads, seq_len, cache_len+seq_len]
        # values:[batch, n_local_heads, cache_len + seq_len, head_dim]
        # attn_value: (batch, n_local_heads, seq_len, head_dim)
        attn_value = torch.matmul(scores, values)
        # attn_value: [batch, n_local_heads, seq_len, head_dim]
        #          => [batch, seq_len, n_local_heads, head_dim]
        #          => [batch, seq_len, n_local_heads* head_dim]
        attn_value = attn_value.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # output: [batch, seq_len, dim]
        output = self.wo(attn_value)
        return output

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        intermediate_size = int(2 * intermediate_size / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            intermediate_size = int(ffn_dim_multiplier * intermediate_size)
        # multiple_of:保证intermediate_size必须是multiple_of的整数倍
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        #
        """
        Y = XA + b, 对A进行 列分块并行,Y=XA+b, 对A各行进行拆分
          A = | A_1, A_2,..., A_p |
        fairscale现已被fsdp替代  
        """
        # self.w1 = ColumnParallelLinear(
        #     dim, intermediate_size, bias=False, gather_output=False, init_method=lambda x: x
        # )
        self.w1 = Linear(
            dim, intermediate_size, bias=False,
        )

        #
        """
        Y = XA + b, 对A进行 行分块并行,Y=XA+b, 对A各行进行拆分
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
        """
        # self.w2 = RowParallelLinear(
        #     intermediate_size, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        # self.w3 = ColumnParallelLinear(
        #     dim, intermediate_size, bias=False, gather_output=False, init_method=lambda x: x
        # )
        self.w2 = Linear(
            intermediate_size, dim, bias=False
        )
        self.w3 = Linear(
            dim, intermediate_size, bias=False
        )

    def forward(self, x):
        # 现在激活函数在gate中
        gate = F.silu(self.w1(x))
        return self.w2(gate * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            intermediate_size=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int, # 推理时kv cache使用
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # freqs_cis:[seq_length, embed_dim//2]
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        # out: [batch, seq_len, dim]
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # tok_embeddings:[vocab_size, embed_size]
        #self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)
        self.tok_embeddings = Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()

        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # 列并行Linear,Y=XA+b, 对A各列进行拆分
        #self.lm_head = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)
        self.lm_head = Linear(params.dim, params.vocab_size, bias=False)

        # freqs_cis: [seq_len, dim // 2]
        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    # 注意：这里只有推理阶段的代码
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        batch, seqlen = tokens.shape
        # h: [batch, seqlen, embed_size]
        h = self.tok_embeddings(tokens)
        # self.freqs_cis: [max_seq_len, dim // 2]
        self.freqs_cis = self.freqs_cis.to(h.device)
        # freqs_cis: [seq_len, dim // 2]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        if seqlen > 1: # 第一次生成prompt prefill时，seqlen>1，其它时都是seqlen=1,因为后面每次只输入一个token, mask均为None
            mask = torch.full((seqlen, seqlen), fill_value=float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1) # 下三角置0, 对角线与下三角均置0

            # 推理时，只计算非mask部分attention,mask部分attention不计算
            # 即mask为:[seqlen, cache_len+seqlen], 而不是[cache_len+seq_len]
            #
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            #
            # hstack进行水平concat
            mask_for_cache = torch.zeros((seqlen, start_pos), device=tokens.device)
            mask = torch.hstack([mask_for_cache, mask]).type_as(h)
        else:
            # 在推理阶段，过了prompt prefilling后， 后面每次只输入一个token, mask均为None
            mask = None

        # h: [batch, seq_len, dim]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # 最后进行一次rms_norm
        # h: [batch, seq_len, dim]
        h = self.norm(h)
        # logits: [batch, seq_len, vocab_size]
        logits = self.lm_head(h).float()
        return logits
