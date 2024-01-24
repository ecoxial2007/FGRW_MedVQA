import torch
from torch import einsum, nn
import torch.nn.functional as F
from collections import OrderedDict
import math


from transformers.activations import gelu


def default(val, d):
    return val if exists(val) else d



class AffineTransform(nn.Module):
    """Do affine Transform for modulating attention score in PyTorch."""

    def __init__(self):
        super(AffineTransform, self).__init__()
        # 定义可训练的参数
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 应用仿射变换
        return x * torch.sigmoid(self.scale) * 5 + self.bias

# normalization
# they use layernorm without bias, something that pytorch does not offer
# Borrow code from:
# https://github.com/lucidrains/CoCa-pytorch
# https://arxiv.org/abs/2205.01917

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask)


    def forward(self, q, k, v, prob=1.0):
        if prob is not None:
            context, attn = self.attention(self.ln_1(q), self.ln_1(k * prob), self.ln_1(v * prob))
        else:
            context, attn = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + context
        q = q + self.mlp(self.ln_2(q))
        return q


def get_mask(lengths, max_length):
    """ Computes a batch of padding masks given batched lengths """
    mask = 1 * (
        torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths
    ).transpose(0, 1)
    return mask

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # dim = config.dim
        dim = config.n_bboxes*config.n_bboxes
        # assert config.dim % config.n_heads == 0
        assert dim % config.n_et_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim))
        ]))

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        # Self-Attention
        sa_output = self.attention(
            query=self.sa_layer_norm(x),
            key=self.sa_layer_norm(x),
            value=self.sa_layer_norm(x),
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )


        sa_output = sa_output[0]
        sa_output = self.output_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network

        ffn_output = x + self.mlp(sa_output)

        output = (ffn_output,)

        return output



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        #print(self.gamma, self.beta, x)
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_et_heads  # config.n_heads
        self.dim = config.n_bboxes * config.n_bboxes  # config.dim
        dp_rate = 0.1  # config.attention_dropout
        self.dropout = nn.Dropout(p=dp_rate)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        dropout, dim, hidden_dim = 0.1, config.n_bboxes*config.n_bboxes, config.d_input
        activation = 'gelu'

        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        assert activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(activation)
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

