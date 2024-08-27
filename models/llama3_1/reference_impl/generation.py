# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from termcolor import cprint

from ..api.args import ModelArgs
from ..api.chat_format import ChatFormat, ModelInput
from ..api.datatypes import CompletionMessage, Message, StopReason
from ..api.tokenizer import Tokenizer
from .model import Transformer


@dataclass
class CompletionPrediction:
    generation: str
    decoded_tokens: Optional[List[str]] = None
    logprobs: Optional[List[List[float]]] = None


@dataclass
class ChatPrediction:
    generation: CompletionMessage
    decoded_tokens: Optional[List[str]] = None
    logprobs: Optional[List[List[float]]] = None


@dataclass
class TokenResult:
    token_id: Optional[int] = None
    text: str = None
    logprobs: Optional[List[float]] = None

@dataclass
class BatchTokenResult:
    token_id: Optional[List[int]] = None
    text: List[str] = None
    logprobs: Optional[List[List[float]]] = None

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.


        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        # 其它gpu不打印日志
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(checkpoints), \
            f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"

        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words

        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        transformer_model = Transformer(model_args)
        transformer_model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        llama = Llama(transformer_model, tokenizer, model_args)
        return llama

    def __init__(self, model: Transformer, tokenizer: Tokenizer, args: ModelArgs):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        model_input: ModelInput,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
    ) -> Generator:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params

        # cprint("Input to model -> " + self.tokenizer.decode(model_input.tokens), "red")
        # prompt_tokens:List[List[int]]
        if model_input.tokens is not None:
            prompt_tokens = [model_input.tokens]
            batch = 1
        else:
            prompt_tokens = model_input.batch_tokens
            batch = len(prompt_tokens)

        assert batch <= params.max_batch_size, (batch, params.max_batch_size)

        # prompt_tokens:List[List[int]]
        # 最长的prompt
        # 最短的prompt
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        if max_prompt_len >= params.max_seq_len:
            cprint(f"Out of token budget {max_prompt_len} vs {params.max_seq_len}", "red" )
            return

        # 生成 + prompt必须小于max_seq_len
        total_len = min(max_gen_len + max_prompt_len, params.max_seq_len)
        pad_id = self.tokenizer.pad_id
        #tokens = torch.full((batch, total_len), fill_value=pad_id, dtype=torch.long, device="cuda")
        tokens = torch.full((batch, total_len), fill_value=pad_id, dtype=torch.long, device="cpu")

        # 将prompt copy到tokens中去
        # prompt_tokens:List[List[int]], shape:[batch, seq_len]
        for text_index, text_tokens in enumerate(prompt_tokens):
            #tokens[text_index, : len(text_tokens)] = torch.tensor(data=text_tokens, dtype=torch.long, device="cuda")
            tokens[text_index, : len(text_tokens)] = torch.tensor(data=text_tokens, dtype=torch.long, device="cpu")

        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        #eos_reached = torch.tensor([False] * batch, device="cuda")
        eos_reached = torch.tensor([False] * batch, device="cpu") # shape:[batch]
        # input_text_mask.shape: [batch, seq_len]
        # input_text_mask==1,为prompt值
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        # List[int]
        stop_tokens = torch.tensor(self.tokenizer.stop_tokens)

        # 从最短的prompt开始生成, 那些比较长的prompt的地方生成的token丢弃不用
        for cur_pos in range(min_prompt_len, total_len):
            # logits:[batch, seq_len, vocab_size]
            logits = self.model.forward(tokens=tokens[:, prev_pos:cur_pos], start_pos=prev_pos) # 将prompt+ 已生成的token直接copy进去

            if temperature > 0:
                # 取所有batch的最后的一个token的logits,即seq_len=-1
                # probs:[batch, vocab_size]
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # next_token:[batch, 1]
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # next_token:[batch, 1]
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            target = tokens[:, prev_pos + 1 : cur_pos + 1]
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )

            # 如果是生成的token，且在stop_tokens里
            eos_reached |= (~input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_tokens))
            if batch==1:
                yield TokenResult(
                    token_id=next_token[0].item(),
                    text=self.tokenizer.decode(next_token.tolist()),
                    logprobs=(
                        token_logprobs[:, prev_pos + 1 : cur_pos + 1][0].tolist() if logprobs else None
                    ),
                )
            else:
                yield BatchTokenResult(
                    token_id=next_token.tolist(),
                    text=[self.tokenizer.decode([x]) for x in next_token.tolist()],
                    logprobs=(
                        token_logprobs[:, prev_pos + 1 : cur_pos + 1].tolist() if logprobs else None
                    ),
                )

            prev_pos = cur_pos
            # 所有的都到了eos,退出
            if all(eos_reached):
                break

    def text_completion(
        self,
        prompt: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> CompletionPrediction:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if ( max_gen_len is None
            or max_gen_len == 0
            or max_gen_len >= self.model.params.max_seq_len
        ):
            max_gen_len = self.model.params.max_seq_len - 1

        # prompt_tokens: List[int]
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)

        token_ids = []
        token_logprobs = []
        decoded_tokens = []
        for result in self.generate( model_input=ModelInput(tokens=prompt_tokens), max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, logprobs=logprobs,):
            token_ids.append(result.token_id)
            if logprobs:
                decoded_tokens.append(result.text)
                token_logprobs.append(result.logprobs)

        generation_tokens = self.tokenizer.decode(token_ids)
        if logprobs:
            return CompletionPrediction(generation=generation_tokens, logprobs=token_logprobs, decoded_tokens=decoded_tokens, )

        return CompletionPrediction(generation=generation_tokens)

    def chat_completion(
        self,
        messages: List[Message],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> ChatPrediction:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if (
            max_gen_len is None
            or max_gen_len == 0
            or max_gen_len >= self.model.params.max_seq_len
        ):
            max_gen_len = self.model.params.max_seq_len - 1

        token_ids = []
        token_logprobs = []
        decoded_tokens = []

        stop_reason = None
        for result in self.generate(model_input=self.formatter.encode_dialog_prompt(messages), max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        ):
            token_ids.append(result.token_id)
            if result.text == "<|eot_id|>":
                stop_reason = StopReason.end_of_turn
            elif result.text == "<|eom_id|>":
                stop_reason = StopReason.end_of_message

            if logprobs:
                decoded_tokens.append(result.text)
                token_logprobs.append(result.logprobs)

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        message = self.formatter.decode_assistant_message(token_ids, stop_reason)

        if logprobs:
            return ChatPrediction( generation=message, logprobs=token_logprobs, decoded_tokens=decoded_tokens, )

        return ChatPrediction(generation=message)


def sample_top_p(probs:torch.Tensor, p:float):
    """
    Top-p 核采样
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    # probs:[batch, vocab_size]
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # probs:[batch, vocab_size]
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # probs_sum-probs_sort就是去除当前位置的累积和
    # 其实就是probs_sum>p, 唯一的区别就是去除当前位置是否>p
    mask = probs_sum - probs_sort > p
    # mask为true的地方都是后面的小概率
    probs_sort[mask] = 0.0
    # probs_sort:[batch, vocab_size], 重新归一化
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # next_token:[batch], 多项式采样
    rand_index = torch.multinomial(probs_sort, num_samples=1)
    # probs_idx:[batch, vocab_size]
    # next_token:[batch]
    next_token = torch.gather(input=probs_idx, dim=-1, index=rand_index)
    return next_token
