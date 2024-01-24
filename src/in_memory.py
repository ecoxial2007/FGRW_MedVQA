import json
import faiss
from tqdm import tqdm
import time
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from perceiver_pytorch import Perceiver
from einops import rearrange, repeat
from utils.attn import ResidualCrossAttentionBlock
from utils.attn import AffineTransform
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import multiprocessing


def process_file(feature_path, mmap_vkeys, mmap_ckeys, mmap_values, index):
    image_feature = h5py.File(feature_path, 'r')
    caption_feature = h5py.File(feature_path.replace('features', 'text_features'), 'r')

    image_global_feature = np.array(image_feature['feature'])
    caption_global_feature = np.array(caption_feature['feature_global'])

    image_local_feature = np.array(image_feature['feature_noproj'])
    caption_local_feature = np.array(caption_feature['feature_local'])
    value_local_feature = np.concatenate([image_local_feature, caption_local_feature], axis=1)


    mmap_vkeys[index] = normalize(image_global_feature, axis=1).squeeze(axis=0)
    mmap_ckeys[index] = normalize(caption_global_feature, axis=1).squeeze(axis=0)
    mmap_values[index] = value_local_feature.squeeze(axis=0)



def process_files_batch(args):
    feature_paths_batch, mmap_vkeys, mmap_ckeys, mmap_values, start_index = args
    for i, feature_path in enumerate(feature_paths_batch):
        process_file(feature_path, mmap_vkeys, mmap_ckeys, mmap_values, start_index + i)



class FeatureMemory(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.device = device
        self.compress_value = Perceiver(
            input_channels=config.d_input,
            input_axis=1,
            num_freq_bands=6,
            max_freq=1.,
            depth=1,
            num_latents=config.compressed_size,
            latent_dim=config.d_input,
            attn_dropout=config.ca_dropout,
            self_per_cross_attn=2,
            final_classifier_head=False
        )
        self.method = config.method
        self.fusion_encoder = ResidualCrossAttentionBlock(config.d_input, config.n_ca_heads, config.ca_dropout)
        self.att_transform = AffineTransform()
        self.memo_path = config.memo_path

    def initial_memory(self):
        start = time.time()
        with open(self.memo_path, 'r') as f:
            feature_paths = json.load(f)

        self.memory_size = len(feature_paths)
        num_processes = multiprocessing.cpu_count()
        chunk_size = len(feature_paths) // num_processes

        feature_paths_batches = [feature_paths[i:i + chunk_size] for i in range(0, len(feature_paths), chunk_size)]

        # 创建内存映射文件
        # 假设的特征维度，需要根据您的数据调整
        feature_dim_vkeys = self.config.d_output
        feature_dim_ckeys = self.config.d_output
        feature_dim_values = self.config.d_input
        if not os.path.exists('./temp/'):
            os.mkdir('./temp/')
        try:
            mmap_vkeys = np.memmap('./temp/mmap_vkeys.dat', dtype='float32', mode='r', shape=(self.memory_size, feature_dim_vkeys))
            mmap_ckeys = np.memmap('./temp/mmap_ckeys.dat', dtype='float32', mode='r', shape=(self.memory_size, feature_dim_ckeys))
            mmap_values = np.memmap('./temp/mmap_values.dat', dtype='float32', mode='r', shape=(self.memory_size, 274, feature_dim_values))
        except:
            mmap_vkeys = np.memmap('./temp/mmap_vkeys.dat', dtype='float32', mode='w+', shape=(self.memory_size, feature_dim_vkeys))
            mmap_ckeys = np.memmap('./temp/mmap_ckeys.dat', dtype='float32', mode='w+', shape=(self.memory_size, feature_dim_ckeys))
            mmap_values = np.memmap('./temp/mmap_values.dat', dtype='float32', mode='w+', shape=(self.memory_size, 274, feature_dim_values))

            for i, batch in tqdm(enumerate(feature_paths_batches), total=len(feature_paths_batches), desc="H5 Loading schedule"):
                process_files_batch((batch, mmap_vkeys, mmap_ckeys, mmap_values, i * chunk_size))

            # # 使用多进程并行处理
            # with multiprocessing.Pool(num_processes) as pool:
            #     pool.map(process_files_batch, [(batch, mmap_vkeys, mmap_ckeys, mmap_values, i * chunk_size) for i, batch in
            #                                    enumerate(feature_paths_batches)])

        print('Finished Loading H5 files: ', time.time() - start)
        print(mmap_vkeys.shape)
        # 使用内存映射文件进行后续处理
        self.vkeys_np = mmap_vkeys
        self.ckeys_np = mmap_ckeys


        d_v = self.vkeys_np.shape[1]
        d_c = self.ckeys_np.shape[1]
        quantizer_v = faiss.IndexFlatIP(d_v)
        quantizer_c = faiss.IndexFlatIP(d_c)

        nlist = 100

        # 使用量化器创建 IndexIVFFlat 索引
        self.vindex = faiss.IndexIVFFlat(quantizer_v, d_v, nlist, faiss.METRIC_INNER_PRODUCT)
        self.cindex = faiss.IndexIVFFlat(quantizer_c, d_c, nlist, faiss.METRIC_INNER_PRODUCT)

        # 训练索引
        if not self.vindex.is_trained:
            self.vindex.train(self.vkeys_np)
        if not self.cindex.is_trained:
            self.cindex.train(self.ckeys_np)

        # 添加向量到索引
        self.vindex.add(self.vkeys_np)
        self.cindex.add(self.ckeys_np)
        print('Finished Initialize Faiss Key:', time.time() - start)

        self.values = mmap_values
        print('Finished Initialize Value:', time.time() - start)
        print('Multi-process Time:', time.time() - start)



    def initial_memory_orgin(self):
        import time
        start = time.time()
        with open(self.memo_path, 'r') as f:
            feature_paths = json.load(f)

        vkey_features = []
        ckey_features = []
        value_features = []
        self.memory_size = len(feature_paths)

        for i, feature_path in enumerate(feature_paths):
            image_feature = h5py.File(feature_path, 'r')
            caption_feature = h5py.File(feature_path.replace('features', 'text_features'), 'r')

            image_global_feature = np.array(image_feature['feature'])
            caption_global_feature = np.array(caption_feature['feature_global'])

            image_local_feature = np.array(image_feature['feature_noproj'])
            caption_local_feature = np.array(caption_feature['feature_local'])
            value_local_feature = np.concatenate([image_local_feature, caption_local_feature], axis=1)

            vkey_features.append(image_global_feature)
            ckey_features.append(caption_global_feature)
            value_features.append(value_local_feature)


        self.vkeys_np = np.concatenate(vkey_features, axis=0)
        self.ckeys_np = np.concatenate(ckey_features, axis=0)
        self.vkeys_np = self.vkeys_np / np.linalg.norm(self.vkeys_np, axis=1, keepdims=True)
        self.ckeys_np = self.ckeys_np / np.linalg.norm(self.ckeys_np, axis=1, keepdims=True)

        d_v = self.vkeys_np.shape[1]
        d_c = self.ckeys_np.shape[1]
        self.vindex = faiss.IndexFlatIP(d_v)
        self.cindex = faiss.IndexFlatIP(d_c)
        self.vindex.add(self.vkeys_np)
        self.cindex.add(self.ckeys_np)
        print('Finished Initialize Faiss Key')

        self.values = torch.from_numpy(np.concatenate(value_features, axis=0))
        print('Finished Initialize Value')
        print('Single Thread Time:', time.time()-start)



    def max_sum_dist(self, query, keys):
        similarity_scores = torch.einsum('bpd,bktd->bkpt', F.normalize(query, dim=-1), F.normalize(keys, dim=-1))

        max_scores, _ = torch.max(similarity_scores, dim=3)
        final_scores = F.softmax(torch.sum(max_scores, dim=2), dim=1)

        return final_scores.unsqueeze(-1).unsqueeze(-1)



    def fusion4answer(self, query_features, top_k_dict):
        top_k_score = self.max_sum_dist(query_features, top_k_dict['top_k_values'])
        top_k_score = top_k_score.expand(-1, -1, self.config.compressed_size, -1)

        top_k_values = rearrange(top_k_dict['top_k_values'], 'b k t d -> (k t) b d')
        top_k_score = rearrange(top_k_score, 'b k t d -> (k t) b d')
        query_features = rearrange(query_features, 'b t d -> t b d')
        fused_query = self.fusion_encoder(query_features, top_k_values, top_k_values, top_k_score)
        return fused_query



    def retrieve_by_faiss(self, vquery, cquery, k):
        vquery_np = vquery.numpy() if isinstance(vquery, torch.Tensor) else vquery
        cquery_np = cquery.numpy() if isinstance(cquery, torch.Tensor) else cquery

        v_D, v_I = self.vindex.search(vquery_np, k)
        c_D, c_I = self.cindex.search(cquery_np, k)
        return v_D, c_D, v_I, c_I

    def retrieve_and_process(self, vquery, cquery):
        """
        根据查询特征检索并处理局部特征。

        :param query_feature: 查询特征。
        :return: 处理后的特征。
        """
        vquery = F.normalize(vquery, p=2, dim=1).cpu()
        cquery = F.normalize(cquery, p=2, dim=1).cpu()
        batch_size = vquery.shape[0]

        v_similarities, c_similarities, v_top_k_indices, c_top_k_indices = self.retrieve_by_faiss(vquery, cquery, self.top_k)

        if 'image' in self.method:
            top_k_similarities = torch.from_numpy(v_similarities).to(self.device)
            top_k_indices = v_top_k_indices
        elif 'text' in self.method:
            top_k_similarities = torch.from_numpy(c_similarities).to(self.device)
            top_k_indices = c_top_k_indices
        else:
            top_2k_similarities = np.concatenate([v_similarities, c_similarities], axis=1)
            top_2k_indices = np.concatenate([v_top_k_indices, c_top_k_indices], axis=1)

            sorted_indices = np.argsort(-top_2k_similarities, axis=1)[:, :self.top_k]
            top_k_similarities = torch.from_numpy(np.take_along_axis(top_2k_similarities, sorted_indices, axis=1)).to(self.device)
            top_k_indices = np.take_along_axis(top_2k_indices, sorted_indices, axis=1)


        top_k_values = np.take(self.values, top_k_indices, axis=0)
        top_k_values = rearrange(torch.from_numpy(top_k_values), 'b k c d -> (b k) c d')
        top_k_values = top_k_values.to(self.device)
        compressed_values = self.compress_value(top_k_values)
        compressed_values = rearrange(compressed_values, '(b k) c d -> b k c d', b=batch_size)


        return {
            'top_k_scores': top_k_similarities,
            'top_k_indices': top_k_indices,
            'top_k_values': compressed_values
        }

    def get_document(self, doc_id):
        return self.document_features(doc_id)


