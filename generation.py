import sys
import io
from pathlib import Path
import argparse
import glob
import utils
import os
import time
from transformers import AutoTokenizer
import jax
import jax.numpy as jnp

from model import LLAMA

if __name__ == "__main__":
    # Initiate the sys for the Korean Encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

    # Argument Parser
    parser = argparse.ArgumentParser(description='Inference with JAX 🚀!!!')
    parser.add_argument("--model_size", type=str, default="jamo_200m")
    parser.add_argument("--model_path", type=str, default="./tmp/model.pt")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model, params = LLAMA.from_pretrained(args.model_size, str(model_path))

    key = jax.random.PRNGKey(0)
    idx = jnp.ones((1, 10), dtype=jnp.int32)
    if args.info:
        print(model.tabulate(key, idx, train=False, depth=1))

    # Loading the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    print("⭐️ Loading LLM Done! ⭐️")

    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    EOS_ID = tokenizer.encode(EOS_TOKEN)[0]

    chat_parser = (
        "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 응답:\n"
    )

    @jax.jit
    def _sample(params, key, tokens) -> jax.Array:
        # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        result = model.generate_jit(
            key, params, tokens, max_new_tokens=10, top_k=5, temperature=0.8
        )
            # result.block_until_read()
        return result

    def sample(params, key, tokens) -> str:
        token = _sample(params, key, tokens)
        target = tokenizer.decode(token[0])
        return target, token

    # JIT 
    sample(params, key, jnp.ones((1, 1), dtype=jnp.int32))

    user_prompt = input(">>> ")

    if args.chat: user_prompt = chat_parser.format_map({"instruction":user_prompt})
    user_prompt = f"{SOS_TOKEN} {user_prompt}"

    idx = tokenizer.encode(user_prompt)
    token = jnp.array([idx], dtype=jnp.int32)   
    print(token)

    num_samples=10
    start = time.time()
    for k in range(num_samples):
        step_key = jax.random.fold_in(key, k)
        sampled_str, token= sample(params, key, token)

    print(sampled_str)
    print(f"Finished in {time.time() - start}")