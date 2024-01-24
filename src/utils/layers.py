import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedEncoder(nn.Module):
    """T5 encoder as a separate model which fuse multi-modal input."""

    def __init__(self, emb_dim, num_heads=8, num_fusion_layers=2, head_dim=768, mlp_dim=768, dropout_rate=0.1, dtype=torch.float32, logits_via_embedding=False):
        super(FusedEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_fusion_layers = num_fusion_layers
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.dtype = dtype
        self.mlp_activations = ('gelu', 'linear')
        self.logits_via_embedding = logits_via_embedding

    def forward(self, fused_input_embs, encoder_input_embs, encoder_mask=None, fused_mask=None, att_mask=None, use_dropout=True, output=False):
        """Function to fuse text and image embedding."""

        if encoder_input_embs is not None:
            x = torch.cat([encoder_input_embs, fused_input_embs], dim=1)
        else:
            x = fused_input_embs

        from utils.t5_pytorch import T5RelativePositionBias
        rel_emb = T5RelativePositionBias(
            scale=1,
            num_buckets=32,
            max_distance=128,
            heads=self.num_heads,
        )
        print(rel_emb.shape)




        if encoder_mask is not None:
            if fused_mask is None:
                pad_width = fused_input_embs.shape[1]
                fused_mask = F.pad(encoder_mask, (0, pad_width), "constant", value=1.0)
            else:
                fused_mask = torch.cat([encoder_mask, fused_mask], dim=1)

        print(fused_mask.shape)



        if output:
            x = nn.LayerNorm(self.emb_dim)(x)

        if att_mask is not None:
            x = x * att_mask

        if use_dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # return the processed tensor and other necessary outputs
        return x  # Add any other required outputs

# Example usage:
# model = FusedT5Encoder(vocab_size=..., emb_dim=..., ...)
