# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from lib.prroi_pool.functional import prroi_pool2d


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, fpn_bn=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, use_fpn_norm=fpn_bn)
        self.encoder_k = base_encoder(num_classes=dim, use_fpn_norm=fpn_bn)

        '''
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        '''

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q=None, im_k=None, ac_q=None, ac_k=None, ac_q_c3=None, ac_k_c3=None, encode_only=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if encode_only:
            return self.encoder_q(im_q, return_mocodet_feats=False)  # queries: NxC

        batch_im_idx = torch.arange(im_q.size(0)).type_as(ac_q)
        batch_idx = batch_im_idx.unsqueeze(1).repeat(1, ac_q.size(1)).unsqueeze(2)  # (N, 12*12, 1)
        batch_idx_c3 = batch_im_idx.unsqueeze(1).repeat(1, ac_q_c3.size(1)).unsqueeze(2)  # (N, 12*12, 1)

        # compute query features
        q, featmap_q, featmap_q_c3 = self.encoder_q(im_q, return_mocodet_feats=True)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, featmap_k, featmap_k_c3 = self.encoder_k(im_k, return_mocodet_feats=True)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            featmap_k = self._batch_unshuffle_ddp(featmap_k, idx_unshuffle)
            featmap_k_c3 = self._batch_unshuffle_ddp(featmap_k_c3, idx_unshuffle)

            # c4 feats
            pooled_featmap_k = nn.functional.avg_pool2d(featmap_k, 3, stride=1, padding=0)  # (N, 128, 12, 12)
            pooled_featmap_k = nn.functional.normalize(pooled_featmap_k, dim=1)
            pooled_featmap_k = concat_all_gather(pooled_featmap_k)
            pooled_featmap_k = pooled_featmap_k.transpose(1, 0).contiguous().view(256, -1)  # (128, Nx12x12)

            # c3 feats
            pooled_featmap_k_c3 = nn.functional.avg_pool2d(featmap_k_c3, 3, stride=1, padding=0)  # (N, 128, 12, 12)
            pooled_featmap_k_c3 = nn.functional.normalize(pooled_featmap_k_c3, dim=1)
            pooled_featmap_k_c3 = pooled_featmap_k_c3.transpose(1, 0).contiguous().view(256, -1)  # (128, Nx12x12)

        # === C4 ===
        ac_q = torch.cat([batch_idx, ac_q], dim=2)
        ac_k = torch.cat([batch_idx, ac_k], dim=2)
        ac_q = ac_q.view(-1, 5)
        ac_k = ac_k.view(-1, 5)

        feat_ac_q = prroi_pool2d(featmap_q, ac_q.detach(), 1, 1, 1/16.)  # (N*12*12, 128, 1, 1)
        with torch.no_grad():
            feat_ac_k = prroi_pool2d(featmap_k, ac_k, 1, 1, 1/16.)  # (N*12*12, 128, 1, 1)

        feat_ac_q = feat_ac_q.squeeze(3).squeeze(2)
        feat_ac_k = feat_ac_k.squeeze(3).squeeze(2)
        feat_ac_q = nn.functional.normalize(feat_ac_q, dim=1)
        feat_ac_k = nn.functional.normalize(feat_ac_k, dim=1)
        # compute logits for moco detection
        ld_pos = torch.einsum('nc,nc->n', [feat_ac_q, feat_ac_k]).unsqueeze(-1)  # Nx1
        ld_neg = torch.einsum('nc,ck->nk', [feat_ac_q, pooled_featmap_k])  # N x N*12*12
        ld_logits = torch.cat([ld_pos, ld_neg], dim=1)
        ld_logits /= self.T

        # === C3 ===
        # C3 anchors
        ac_q_c3 = torch.cat([batch_idx_c3, ac_q_c3], dim=2)
        ac_k_c3 = torch.cat([batch_idx_c3, ac_k_c3], dim=2)
        ac_q_c3 = ac_q_c3.view(-1, 5)
        ac_k_c3 = ac_k_c3.view(-1, 5)

        feat_ac_q_c3 = prroi_pool2d(featmap_q_c3, ac_q_c3.detach(), 1, 1, 1/8.)  # (N*12*12, 128, 1, 1)
        with torch.no_grad():
            feat_ac_k_c3 = prroi_pool2d(featmap_k_c3, ac_k_c3, 1, 1, 1/8.)  # (N*12*12, 128, 1, 1)

        feat_ac_q_c3 = feat_ac_q_c3.squeeze(3).squeeze(2)
        feat_ac_k_c3 = feat_ac_k_c3.squeeze(3).squeeze(2)
        feat_ac_q_c3 = nn.functional.normalize(feat_ac_q_c3, dim=1)
        feat_ac_k_c3 = nn.functional.normalize(feat_ac_k_c3, dim=1)
        # compute logits for moco detection
        ld_pos_c3 = torch.einsum('nc,nc->n', [feat_ac_q_c3, feat_ac_k_c3]).unsqueeze(-1)  # Nx1
        ld_neg_c3 = torch.einsum('nc,ck->nk', [feat_ac_q_c3, pooled_featmap_k_c3])  # N x N*12*12
        ld_logits_c3 = torch.cat([ld_pos_c3, ld_neg_c3], dim=1)
        ld_logits_c3 /= self.T        

        # === INSTANCE ===
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return (logits, labels), ld_logits, ld_logits_c3


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
