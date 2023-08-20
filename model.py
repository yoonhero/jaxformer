from typing import Any
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.core import freeze
from flax.traverse_util import path_aware_map
from flax.training import train_state
from flax import traverse_util

from dataclasses import dataclass
from typing import Optional, Any, Tuple
from functools import partial

@dataclass
class LLAMAConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int = 512
    multiple_of: int=256
    dropout: float=0.0


class CasualSelfAttention(nn.Module):
    config: LLAMAConfig 

    def setup(self):
        config = self.config
        assert config.n_embd % config.n_head == 0
        self.head_size = config.n_embd // config.n_head

        self.c_attn = nn.Dense(config.n_embd * 3, use_bias=False)
        self.c_proj = nn.Dense(config.n_embd, use_bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x: jax.Array, *, mask: Optional[jax.Array], train: bool) -> jax.Array:
        B, T, C = x.shape

        q, k, v = self.c_attn(x)._split(3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        k = k.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)
        v = v.reshape(B, T, self.n_head, self.head_size).swapaxes(1, 2)

        # mask = jnp.tril(jnp.ones((T, T))).reshape((1,1,T,T))
    
        att = (q @ k.swapaxes(-2, -1)) * (1.0/jnp.sqrt(k.shape[-1]))
        # att = jnp.where(mask==0, float('-inf'), att)
        if mask is not None:
            att = att + mask

        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v
        y = y.swapaxes(1, 2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y), deterministic=not train)
        return y


class MLP(nn.Module):
    config:LLAMAConfig 

    def setup(self):
        config = self.config

        dim = config.n_embd
        hidden_dim = dim * 4
        multiple_of = config.multiple_of

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1)) // multiple_of

        self.w1 = nn.Dense(hidden_dim, use_bias=False)
        self.w2 = nn.Dense(dim, use_bias=False)
        self.w3 = nn.Dense(hidden_dim, use_bias=False)
        
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class FeedForward(nn.Module):
    config: LLAMAConfig

    def setup(self):
        config = self.config

        self.c_fc = nn.Dense(config.n_embd*4, use_bias=False)
        self.c_proj = nn.Dense(config.n_embd, use_bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def __call__(self, x: jax.Array, *, train: bool):
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=not train)
        return x


class Block(nn.Module):
    config:LLAMAConfig 

    def setup(self):
        config = self.config
        self.ln_1 = nn.RMSNorm(epsilon=1e-5)
        self.ln_2 = nn.RMSNorm(epsilon=1e-5)

        self.attn = CasualSelfAttention(config)
        # self.mlp = MLP(config)
        self.mlp = FeedForward(config)

    def __call__(self, x: jax.Array, *, mask: Optional[jax.Array], train: bool) -> jax.Array:
        x = x + self.attn(self.ln_1(x), mask=mask, train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x

class LLAMA(nn.Module):
    config:LLAMAConfig 

    def setup(self):
        config = self.config

        assert config.vocab_size is not None
        assert config.block_size is not None

        self.wte = nn.Embed(config.vocab_size, config.n_embd)
        self.wpe = nn.Embed(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.projection = nn.Dense(config.vocab_size, use_bias=False)
    
    def __call__(self, idx: jax.Array, *, train: bool,  mask: Optional[jax.Array]=None, targets: Optional[jax.Array]=None):
        b, t = idx.shape
        assert t <= self.config.block_size
        pos = jnp.arange(0, t, dtype=jnp.int32)[None]

        tok_emb = self.wte(idx)
        # TODO: implement RoPE
        # pos_emb = self.wpe(pos)
        # print(pos_emb)
        x = self.drop(tok_emb, deterministic=not train)

        for block in self.h:
            x = block(x, mask=mask, train=train)

        h = self.ln_f(x)
        output = self.projection(h)

        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                output, targets
            ).mean()
        else: loss = None

        return output, loss

    @classmethod
    def from_pretrained(cls, model_type, path):
        assert model_type in {"jamo_200m"}
        
        config_args = {
            "jamo_200m":dict(n_layer=15, n_head=16, n_embd=1024, vocab_size=8000, block_size=256)
        }[model_type]

        config = LLAMAConfig(**config_args)
        model = LLAMA(config)
        variables = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), jnp.ones((1, 1), dtype=jnp.int32), train=False))
        params = variables["params"]
        flat_params = traverse_util.flatten_dict(params, sep=".")

        # model_jamo = JAMO.from_pretrained("small", path, "cpu")
        # sd_jamo = model_jamo.state_dict()
        import torch
        sd_jamo = torch.load(path, map_location=torch.device("cpu"))

        base = "_orig_mod"
        def copy_from(flax_name, pt_name, transpose=False):
            pt_tensor = sd_jamo[f"{base}.{pt_name}"]
            jax_array = flat_params[flax_name]
            if transpose: pt_tensor = pt_tensor.t()
            pt_array = pt_tensor.detach().cpu().numpy()

            assert pt_array.shape == jax_array.shape, "During loading checkpoint source and target shape is not same!"

            flat_params[flax_name] = pt_array

        copy_from("wte.embedding", "transformer.wte.weight")
        copy_from("ln_f.scale", "transformer.ln_f.weight")
        copy_from("ln_f.bias", "transformer.ln_f.bias")
        copy_from("projection.kernel", "lm_head.weight", transpose=True)

        for i in range(config.n_layer):
            ## TODO: implement custom rmsnorm with bias
            #   '_orig_mod.transformer.h.{i}.rms_1.bias', 
            #  '_orig_mod.transformer.h.{i}.rms_2.bias',
            copy_from(f"h_{i}.attn.c_attn.kernel", f"transformer.h.{i}.sa.c_attn.weight", transpose=True)
            copy_from(f"h_{i}.attn.c_proj.kernel", f"transformer.h.{i}.sa.c_proj.weight", transpose=True)
            copy_from(f'h_{i}.ln_1.scale', f"transformer.h.{i}.rms_1.weight")
            copy_from(f'h_{i}.ln_2.scale', f'transformer.h.{i}.rms_2.weight')
            copy_from(f"h_{i}.mlp.c_fc.kernel", f'transformer.h.{i}.mlp.c_fc.weight', transpose=True)
            copy_from(f"h_{i}.mlp.c_proj.kernel", f"transformer.h.{i}.mlp.c_proj.weight", transpose=True)

        params = freeze(traverse_util.unflatten_dict(flat_params, sep="."))

        return model, params

    def configure_optimizers(self, params, weight_decay, learning_rate, betas):
        def get_optimizer(decay):
            return optax.adamw(
                learning_rate=learning_rate, b1=betas[0], b2=betas[1], weight_decay=decay
            )

        def partition_fn(path: Tuple[str, ...], x) -> str:
            if path[-1] in ('bias', 'scale', 'embedding'):
                return "no_decay"
            elif path[-1] in ("kernal", ):
                return "decay"
            else:
                raise ValueError(f"Unrecognized parameter: {path}")

        partition_optimizers = {
            "decay": get_optimizer(weight_decay),
            "no_decay": get_optimizer(0.0)
        }
        param_partitions = freeze(path_aware_map(partition_fn, params))
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        return tx

    # @partial(jax.jit, static_argnames=("self", "length"))
    def generate_jit(self, key, params, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        B, T = input_tokens.shape
        padding = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens = jnp.concatenate([input_tokens, padding], axis=-1)
        indexes = jnp.arange(T, T + max_new_tokens)

        # tokens index -> tokens None
        def scan_f(tokens, i):
            # l: x y
            # t: a b - -
            # i: 0 1 2 3
            step_key = jax.random.fold_in(key, i)
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self.apply({'params': params}, tokens, train=False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, i - 1, :] / temperature
            # optionally crop the logits to only the top k options
            # sample from the distribution
            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
                next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1).squeeze(-1)
            else:
                next_token = jax.random.categorical(step_key, logits, axis=-1)
                # logits = jnp.where(logits < v[:, -1:], float('-inf'), logits)
            # append sampled index to the running sequence and continue
            tokens = tokens.at[:, i].set(next_token)

            return tokens, None

        tokens, _ = jax.lax.scan(scan_f, tokens, indexes)

        return tokens

    def generate(self, random_key, params, context, length, temp=1.0, top_k=None):
        if context.shape[-1] > self.config.block_size:
            context = context[:, -self.config.block_size]
        logits, _ = self.apply({"params":params}, context, train=False)    
        random_key, random_subkey = jax.random.split(random_key)
        new_token = jax.random.categorical(random_subkey, logits[:, -1, :], axis=-1, shape=(1, 1))
        context = jnp.concatenate([context, new_token], axis=1)

        return context

    def create_state(
        self, learning_rate, weight_decay, beta1, beta2,
        decay_lr=None, warmup_iters=None, lr_decay_iters=None, min_lr=None, params=None, **kwargs
    ):
        if params is None:
            variables = self.init(jax.random.PRNGKey(0), jnp.ones((1,1), dtype=jnp.int32), train=False)
            params = variables["params"]
        if decay_lr:
            assert warmup_iters is not None and lr_decay_iters is not None and min_lr is not None
            lr_scheduler = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=learning_rate, warmup_steps=warmup_iters, decay_steps=lr_decay_iters, end_value=min_lr
            )
        else:
            lr_scheduler = learning_rate

        tx = self.configure_optimizers(
            params, weight_decay=weight_decay, learning_rate=lr_scheduler, betas=(beta1, beta2)
        )
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)
        