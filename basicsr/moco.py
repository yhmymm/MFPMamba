import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, projection=True, proj_dim=128, proj='convmlp'):
        super(ProjectionHead, self).__init__()
        self.projection = projection
        if projection:
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )
            print('use projector')
        self.nanchor = 100
        self.nhard_anchor = 50
        self.npos = 300 #300 200
        self.nneg = 500 #500 400
        self.nhard_neg = 100 #100 50
        self.classes = [0,1]

    def _hard_anchor_sampling(self, X, y_hat, y):
        ## 4 256*512 128
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        anchor_f = []
        pos_f = []
        neg_f = []
        ## y  4 256*512
        for ii in range(batch_size):            
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            for cls_id in self.classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if cls_id == 1:
                    num_hard_keep = min(num_hard, self.nhard_anchor)
                    
                    num_easy_keep = num_easy // 3 
                    num_easy_keep = min(num_easy_keep, self.nanchor - self.nhard_anchor)           
                    num_pos_keep = num_easy - num_easy_keep
                    num_pos_keep = min(num_pos_keep, self.npos)

                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    
                    perm = torch.randperm(num_easy)
                    anchor_easy_indices = easy_indices[perm[:num_easy_keep]]
                    pos_indices = easy_indices[perm[num_easy_keep:num_easy_keep+num_pos_keep]]
                    
                    anchor_indices = torch.cat((hard_indices, anchor_easy_indices), dim=0)

                    anchor_f_ = X[ii, anchor_indices, :].squeeze(1)
                    pos_f_ = X[ii, pos_indices, :].squeeze(1)

                    anchor_f.append(anchor_f_)
                    pos_f.append(pos_f_)
                else:
                    num_hard_keep = min(num_hard, self.nhard_neg)
                    num_easy_keep = min(num_easy, self.nneg - num_hard_keep)
                    
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    
                    perm = torch.randperm(num_easy)
                    easy_indices = easy_indices[perm[:num_easy_keep]]
                    indices = torch.cat((hard_indices, easy_indices), dim=0)
                    neg_f_ = X[ii, indices, :].squeeze(1)
                    neg_f.append(neg_f_)

        anchor_x = torch.cat(anchor_f, dim=0)
        pos_x = torch.cat(pos_f, dim=0)
        neg_x = torch.cat(neg_f, dim=0)
        return anchor_x, pos_x, neg_x

    def forward(self, x, labels, predict):
        ## 4 128 256 512
        if self.projection:
            x = self.proj(x)
        x = F.normalize(x, p=2, dim=1)
        ## 4 1 256 512
        ## 4 2 256 512 -> 4 1 256 512

        batch_size = x.shape[0]

        ## 4 256*512
        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        ## 4 256 512 128 
        x = x.permute(0, 2, 3, 1)
        ## 4 256*512 128
        x = x.contiguous().view(x.shape[0], -1, x.shape[-1])

        anchor_x, pos_x, neg_x = self._hard_anchor_sampling(x, labels, predict)
        return anchor_x, pos_x, neg_x

class PixelContrast(nn.Module):
    def __init__(self, use_mem=False, dim=128, K=8000, T=0.1, base_T=0.07):
        super(PixelContrast, self).__init__()

        self.K = K
        self.T = T
        self.base_T = base_T
        self.full = False
        self.use_mem = use_mem
        # create the queue
        if use_mem:
            self.register_buffer("queue", torch.randn(K, dim))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size >= self.K:
            self.full = True
        if ptr + batch_size > self.K:
            tail = self.K - ptr
            self.queue[ptr : ptr + tail, :] = keys[:tail, :]
            head = batch_size - tail
            ptr = 0
            self.queue[ptr : ptr + head, :] = keys[tail:, :]
            ptr = ptr + head
        else:
            self.queue[ptr : ptr + batch_size, :] = keys
            ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, anchor_x, pos_x, neg_x):
        ## batch * nanchor, 128
        ## batch * npos, 128
        ## batch * nneg, 128
        n_pos = pos_x.shape[0]
        # n_neg = neg_x.shape[0]
        ## (np+nn) * 128
        if self.use_mem:
            self._dequeue_and_enqueue(neg_x)
            if self.full:
                neg_x = self.queue
        
        sample_x = torch.cat((pos_x, neg_x), dim=0)
        ## na * (np+nn)
        anchor_dot_samples = torch.div(torch.matmul(anchor_x, sample_x.T), self.T)
        ## na * 1
        logits_max, _ = torch.max(anchor_dot_samples, dim=1, keepdim=True)
        ## na * (np+nn)
        logits = anchor_dot_samples - logits_max.detach()
        ## na * np ;  na * nn
        pos_logits, neg_logits = anchor_dot_samples[:, :n_pos], anchor_dot_samples[:, n_pos:]
        ## na * nn -> na * 1
        neg_logits = torch.exp(neg_logits).sum(1, keepdim=True)
        ## na * np
        exp_pos = torch.exp(pos_logits)
        ## na * np
        logits_prob = pos_logits - torch.log(exp_pos + neg_logits)
        ## na * np -> na * 1
        mean_log_prob_pos = logits_prob.mean(dim=1)
        loss = - (self.T / self.base_T) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss