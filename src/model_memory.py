import torch
from torch import nn
from einops import rearrange, repeat
from utils.attn import ResidualCrossAttentionBlock
from in_memory import FeatureMemory


class ModelConfig:
    def __init__(self, n_ca_heads, ca_dropout, d_input, d_output, compressed_size, method,  top_k, memo_path):
        self.n_ca_heads = n_ca_heads
        self.ca_dropout = ca_dropout
        self.d_input = d_input
        self.d_output = d_output
        self.compressed_size = compressed_size
        self.method = method
        self.top_k = top_k
        self.memo_path = memo_path


    @classmethod
    def from_args(cls, args):
        return cls(n_ca_heads=args.n_ca_heads,
                   ca_dropout=args.ca_dropout,
                   d_input=args.d_input,
                   d_output=args.d_output,
                   compressed_size=args.compressed_size,
                   method=args.method,
                   top_k=args.top_k,
                   memo_path=args.memo_path
                   )



class VQAModel(nn.Module):

    def __init__(self, config: ModelConfig, device):
        super().__init__()
        self.config = config
        self.method = config.method
        self.device = device

        self.mhca_i2t = ResidualCrossAttentionBlock(config.d_input, config.n_ca_heads, config.ca_dropout)
        self.mhca_t2i = ResidualCrossAttentionBlock(config.d_input, config.n_ca_heads, config.ca_dropout)
        self.mhca_m2a = ResidualCrossAttentionBlock(config.d_output, 8, config.ca_dropout)

        self.linear_merge = nn.Linear(config.d_input * 2, config.d_output)
        self.linear_merge_image = nn.Linear(config.d_input, config.d_output)

        self.memo = FeatureMemory(config=config, device=device)
        self.memo.initial_memory()
        self.memo.to(device)


    def forward(self, item_dict):
        iFeature = item_dict['image_features'].squeeze(dim=1)
        qFeature = item_dict['text_query_features']



        pFeature = item_dict['patch_features'].squeeze(dim=1)
        # iFeature = self.memo.compress_query_image(pFeature).squeeze(dim=1).cpu().detach()

        tFeature = item_dict['text_query_token_features']
        # qFeature = self.memo.compress_query_text(tFeature).squeeze(dim=1).cpu().detach()


        if 'memo' in self.method:
            top_k_dict = self.memo.retrieve_and_process(iFeature, qFeature)
            pFeature = self.memo.fusion4answer(pFeature, top_k_dict)
        else:
            pFeature = rearrange(pFeature, 'b n c -> n b c')

        tFeature = rearrange(tFeature, 'b n c -> n b c')
        pFeature_merge = self.mhca_i2t(pFeature, tFeature, tFeature).mean(dim=0)
        tFeature_merge = self.mhca_t2i(tFeature, pFeature, pFeature).mean(dim=0)

        Feature_merge = torch.cat([pFeature_merge, tFeature_merge], dim=-1)
        Feature_merge = self.linear_merge(Feature_merge).unsqueeze(dim=0)
        aFeature = rearrange(item_dict['text_cands_features'], 'b t c -> t b c')
        Feature_merge = self.mhca_m2a(Feature_merge, aFeature, aFeature).mean(dim=0)
        return Feature_merge, top_k_dict['top_k_indices']


