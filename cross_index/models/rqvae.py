import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer

class CrossRQVAE(nn.Module):
    def __init__(self,
                 text_in_dim=768,
                 image_in_dim=768,
                 num_emb_list=[256,256,256,256],
                 e_dim=64,
                 layers=[512,256,128],
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=[0.0,0.0,0.0,0.003], ###### 前三层量化普通量化，只有最后一层开启了 Sinkhorn 算法
                 sk_iters=100,
                 use_linear=0,
                 begin_cross_layer=4,
                 use_cross_rq=False,
                 text_class_info=None,
                 image_class_info=None,
                 text_contrast_weight=1.0,
                 image_contrast_weight=1.0,
                 recon_contrast_weight=0.001,
        ):
        super(CrossRQVAE, self).__init__()

        self.text_in_dim = text_in_dim
        self.image_in_dim = image_in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.num_rq_layers = len(num_emb_list)
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.begin_cross_layer = begin_cross_layer
        self.use_cross_rq = use_cross_rq
        self.text_class_info = text_class_info
        self.image_class_info = image_class_info
        self.align_dim = 768
        self.text_contrast_weight = text_contrast_weight
        self.image_contrast_weight = image_contrast_weight
        self.recon_contrast_weight = recon_contrast_weight
        self.text_encode_layer_dims = [self.align_dim] + self.layers + [self.e_dim]
        self.image_encode_layer_dims = [self.align_dim] + self.layers + [self.e_dim]
       
        self.text_align_encoder = MLPLayers(layers=[self.text_in_dim, self.align_dim],
                                            dropout=self.dropout_prob,bn=self.bn)
        self.image_align_encoder = MLPLayers(layers=[self.image_in_dim, self.align_dim],
                                            dropout=self.dropout_prob,bn=self.bn)
        self.text_align_decoder = MLPLayers(layers=[self.align_dim, self.text_in_dim],
                                            dropout=self.dropout_prob,bn=self.bn)
        self.image_align_decoder = MLPLayers(layers=[self.align_dim, self.image_in_dim],
                                            dropout=self.dropout_prob,bn=self.bn)
        self.text_encoder = MLPLayers(layers=self.text_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.text_rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear)

        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(layers=self.text_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

        # image 
        self.image_encoder = MLPLayers(layers=self.image_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        
        self.image_rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear)

        self.image_decode_layer_dims = self.image_encode_layer_dims[::-1]
        self.image_decoder = MLPLayers(layers=self.image_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    '''
    带类别的 监督学习 跨模态对比学习量化
    同class 的Item 的 Embedding 更接近
    '''
    def class_con_cross_rq(
        self, text_vq, image_vq, residual_text_x, residual_image_x, text_x, image_x, use_sk=True, item_index=None, temperature=0.1
    ):
        text_x_res, text_loss, text_indices, text_distances = text_vq(residual_text_x, use_sk=use_sk)
        image_x_res, image_loss, image_indices, image_distances = image_vq(residual_image_x, use_sk=use_sk)
        if item_index is not None:
            batch_size = residual_text_x.size(0)

            ################## 特征归一化 ##################
            text_feat = F.normalize(residual_text_x, p=2, dim=1)
            image_feat = F.normalize(residual_image_x, p=2, dim=1)

            batch_global2batch_idx = {int(idx): i for i, idx in enumerate(item_index)}

            pos_idx_text = []
            for i in range(batch_size):
                anchor_global_idx = int(item_index[i])
                ##################### 找与当前 text 同类别的 image item #####################
                pos_global_indices = set(self.image_class_info.get(anchor_global_idx, []))
                ##################### 只保留 batch 内的 positive #####################
                batch_pos = [batch_global2batch_idx[gidx] for gidx in pos_global_indices if gidx in batch_global2batch_idx and batch_global2batch_idx[gidx] != i]
                ##################### 随机选一个 positive / 如果没有，自己和自己配对
                if batch_pos:
                    pos = random.choice(batch_pos)
                else:
                    pos = i
                pos_idx_text.append(pos)
            pos_idx_text = torch.tensor(pos_idx_text, device=residual_text_x.device)

            sim_matrix_text = torch.matmul(text_feat, text_feat.T) / temperature  # [B, B]
            pos_sim_text = sim_matrix_text[torch.arange(batch_size), pos_idx_text]
            ##################### InfoNCE Loss #####################
            loss_text = -torch.log(
                torch.exp(pos_sim_text) / torch.exp(sim_matrix_text).sum(dim=1)
            ).mean()

            pos_idx_image = []
            for i in range(batch_size):
                anchor_global_idx = int(item_index[i])
                pos_global_indices = set(self.text_class_info.get(anchor_global_idx, []))
                batch_pos = [batch_global2batch_idx[gidx] for gidx in pos_global_indices if gidx in batch_global2batch_idx and batch_global2batch_idx[gidx] != i]
                if batch_pos:
                    pos = random.choice(batch_pos)
                else:
                    pos = i
                pos_idx_image.append(pos)
            pos_idx_image = torch.tensor(pos_idx_image, device=residual_image_x.device)

            sim_matrix_image = torch.matmul(image_feat, image_feat.T) / temperature  # [B, B]
            pos_sim_image = sim_matrix_image[torch.arange(batch_size), pos_idx_image]
            loss_image = -torch.log(
                torch.exp(pos_sim_image) / torch.exp(sim_matrix_image).sum(dim=1)
            ).mean()

            ##################### 最终Loss=InfoNCE Loss+VQ Loss #####################
            text_loss = text_loss + self.text_contrast_weight * loss_text
            image_loss = image_loss + self.image_contrast_weight * loss_image

        return text_x_res, text_loss, text_indices, text_distances, image_x_res, image_loss, image_indices, image_distances

    def forward(self, text_x, image_x, item_index=None, use_sk=True):
        ##################### 模态对齐【text/image映射到更接近的空间】 #####################
        text_align_in = self.text_align_encoder(text_x)
        image_align_in = self.image_align_encoder(image_x)
        text_x = self.text_encoder(text_align_in)
        image_x = self.image_encoder(image_align_in)
        
        text_rq_loss = []
        image_rq_loss = []
        text_indices_list = []
        image_indices_list = []
        text_distances_list = []
        image_distances_list = []
        text_x_q = 0
        image_x_q = 0
        residual_text_x = text_x
        residual_image_x = image_x
        if self.use_cross_rq:
            for i in range(self.num_rq_layers):
                text_vq = self.text_rq.vq_layers[i]
                image_vq = self.image_rq.vq_layers[i]
                ################### 前几层用普通rq，后面几层开始text/image相互约束 ###################
                if i >= self.begin_cross_layer:
                    ################### 语义对齐 ###################
                    text_x_res, text_loss, text_indices, text_distances, image_x_res, image_loss, image_indices, image_distances = self.class_con_cross_rq(text_vq, image_vq, residual_text_x, residual_image_x, text_x, image_x, use_sk=use_sk, item_index=item_index)
                    residual_text_x = residual_text_x - text_x_res
                    residual_image_x = residual_image_x - image_x_res
                    text_x_q = text_x_q + text_x_res
                    image_x_q = image_x_q + image_x_res
                else:
                    text_x_res, text_loss, text_indices, text_distances = text_vq(residual_text_x, use_sk=use_sk)
                    image_x_res, image_loss, image_indices, image_distances = image_vq(residual_image_x, use_sk=use_sk)
                    residual_text_x = residual_text_x - text_x_res
                    residual_image_x = residual_image_x - image_x_res
                    text_x_q = text_x_q + text_x_res
                    image_x_q = image_x_q + image_x_res
                text_rq_loss.append(text_loss)
                text_indices_list.append(text_indices)
                text_min_distance = text_distances.min(dim=-1)[0] 
                text_distances_list.append(text_min_distance)

                image_rq_loss.append(image_loss)    
                image_indices_list.append(image_indices)
                image_min_distance = image_distances.min(dim=-1)[0] 
                image_distances_list.append(image_min_distance)
            text_rq_loss = torch.stack(text_rq_loss).mean()
            image_rq_loss = torch.stack(image_rq_loss).mean()
            text_indices = torch.stack(text_indices_list, dim=-1)
            image_indices = torch.stack(image_indices_list, dim=-1)
            text_distances = torch.stack(text_distances_list, dim=1)
            image_distances = torch.stack(image_distances_list, dim=1)
        else:
            text_x_q, text_rq_loss, text_indices, text_distances = self.text_rq(text_x, use_sk=use_sk)
            image_x_q, image_rq_loss, image_indices, image_distances = self.image_rq(image_x, use_sk=use_sk)
        text_align_out = self.text_decoder(text_x_q)
        image_align_out = self.image_decoder(image_x_q)
        text_out = self.text_align_decoder(text_align_out)
        image_out = self.image_align_decoder(image_align_out)
        share_out = (text_x_q, image_x_q)

        return text_out, image_out, text_rq_loss, image_rq_loss, text_indices, image_indices, text_distances, image_distances, share_out


    '''
    跨模态对齐损失
    重建后的text/image Embedding对齐
    【reconstruction不一定保证text、image重建语义一致 -> 增加重建后跨模态对齐】
    '''
    def text_image_recon_align(self, text_out, image_out, temperature=0.1):

        text_out_norm = F.normalize(text_out, p=2, dim=1)
        image_out_norm = F.normalize(image_out, p=2, dim=1)
        sim_matrix = torch.matmul(text_out_norm, image_out_norm.T) / temperature  # [batch, batch]
        batch_size = text_out.size(0)
        labels = torch.arange(batch_size, device=text_out.device)
        ############### （1）给定text->找image；（2）给定image->text 双向对齐###############
        loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        return loss
    '''
    重建损失+量化损失+重建后多模态对齐损失
    1) 保证semantic token能恢复原Embedding
    2) 保证latent Embedding能稳定离散化
    3) text/image在共享空间对齐
    '''
    def compute_loss(self, text_out, image_out, text_rq_loss, image_rq_loss, text_indices, image_indices, text_distances, image_distances, text_xs, image_xs, share_out):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(text_out, text_xs, reduction='mean') + F.mse_loss(image_out, image_xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(text_out, text_xs, reduction='mean') + F.l1_loss(image_out, image_xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        align_loss = self.text_image_recon_align(share_out[0], share_out[1])
        loss_total = loss_recon + self.quant_loss_weight * (text_rq_loss + image_rq_loss) + self.recon_contrast_weight * align_loss
        return loss_total, loss_recon
    
    '''
    生成semantic token【semantic ID】
    '''
    @torch.no_grad()
    def get_indices(self, text_xs, image_xs, use_sk=False):
        text_align_in = self.text_align_encoder(text_xs)
        image_align_in = self.image_align_encoder(image_xs)
        text_x_e = self.text_encoder(text_align_in)
        image_x_e = self.image_encoder(image_align_in)
        residual_text_x = text_x_e
        residual_image_x = image_x_e

        if self.use_cross_rq:
            batch_size = text_x_e.size(0)
            text_indices_list = []
            image_indices_list = []
            text_distances_list = [[] for i in range(self.num_rq_layers)]
            image_distances_list = [[] for i in range(self.num_rq_layers)]
            for i in range(self.num_rq_layers):
                text_vq = self.text_rq.vq_layers[i]
                image_vq = self.image_rq.vq_layers[i]
                if i >= self.begin_cross_layer:
                    text_x_res, _, text_indices, text_distances = text_vq(residual_text_x, use_sk=use_sk)
                    image_x_res, _, image_indices, image_distances = image_vq(residual_image_x, use_sk=use_sk)
                else:
                    text_x_res, _, text_indices, text_distances = text_vq(residual_text_x, use_sk=use_sk)
                    image_x_res, _, image_indices, image_distances = image_vq(residual_image_x, use_sk=use_sk)
                residual_text_x = residual_text_x - text_x_res
                residual_image_x = residual_image_x - image_x_res
                text_indices_list.append(text_indices)
                image_indices_list.append(image_indices)
                text_distances_list[i].extend(text_distances.cpu().numpy().tolist())
                image_distances_list[i].extend(image_distances.cpu().numpy().tolist())
            text_indices = torch.stack(text_indices_list, dim=-1)
            image_indices = torch.stack(image_indices_list, dim=-1)
            text_distances = text_distances_list
            image_distances = image_distances_list
        else:
            _, _, text_indices, text_distances = self.text_rq(text_x_e, use_sk=use_sk)
            _, _, image_indices, image_distances = self.image_rq(image_x_e, use_sk=use_sk)
        return text_indices, image_indices, text_distances, image_distances
    
    

'''
单模态RQVAE:
(1) Encoder: 将embedding压缩到latent space
(2) 多层 残差 将连续latent离散为 semantic token sequence
(3) Deocder: 重建原Embedding
'''
class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices, distances = self.rq(x,use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances

    def compute_loss(self, out, quant_loss, xs=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon