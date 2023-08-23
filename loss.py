'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tools import *
import numpy as np


class AAMsoftmax(nn.Module):
    def __init__(self, m, s):

        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, cosine, label):
        # 由 cosθ 计算相应的 sinθ
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        # 展开计算 cos(θ+m) = cosθ*cosm - sinθ*sinm, 其中包含了 Target Logit (cos(θyi+ m))
        # (由于输入特征 xi 的非真实类也参与了计算, 最后计算新 Logit 时需使用 One-Hot 区别)
        phi = cosine * self.cos_m - sine * self.sin_m
        # cos(theta + m) 余弦公式
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot -----------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将 labels 转换为独热编码, 用于区分是否为输入特征 xi 对应的真实类别 yi
        one_hot = torch.zeros_like(cosine, device='cuda')
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # 计算新 Logit
        #  - 只有输入特征 xi 对应的真实类别 yi (one_hot=1) 采用新 Target Logit cos(θ_yi + m)
        #  - 其余并不对应输入特征 xi 的真实类别的类 (one_hot=0) 则仍保持原 Logit cosθ_j
        # 将样本的标签映射为one hot形式 例如N个标签，映射为（N，num_classes）
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # 对于正确类别（1*phi）即公式中的cos(theta + m)，对于错误的类别（1*cosine）即公式中的cos(theta）
        # 这样对于每一个样本，比如[0,0,0,1,0,0]属于第四类，则最终结果为[cosine, cosine, cosine, phi, cosine, cosine]
        # 再乘以半径，经过交叉熵，正好是ArcFace的公式
        output = output * self.s
        return output

class AAMLoss(nn.Module):
    def __init__(self, n_class, margin=0.2, scale=30):
        super(AAMLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce_ulb = nn.CrossEntropyLoss(reduction='none')
        self.ce_lb = nn.CrossEntropyLoss()
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label, thresold=0, labelled=False, gated=True):

        if labelled==False:
            xw, xs = x.chunk(2)
            cosine = F.linear(F.normalize(x), F.normalize(self.weight))

            cosine_w = F.linear(F.normalize(xw), F.normalize(self.weight))
            pseudo_label_w = torch.softmax(cosine_w, dim=-1)
            max_prob_w, max_idx_w = torch.max(pseudo_label_w, dim=-1)

            sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            # max_prob, _ = torch.max(output, dim=-1)
            output = output * self.s

            if gated == True:
                mask_w = max_prob_w > thresold
                ce = self.ce_ulb(output, label)
                # loss_mask = ce <= gate  # Find the sample that loss smaller that gate
                mask_w=torch.cat((mask_w, mask_w.clone()))
                nselect = sum(mask_w).detach()  # Count the num
                # Compute the loss for the selected data only
                unsup_loss = ce * mask_w
                #mask_prob = max_prob_w * mask_w
                # Compute the training acc for these selected data only
                prec_lb = accuracy(output.detach(), label * mask_w.detach(), topk=(1,))[0] * cosine.size()[0]
                return unsup_loss.mean(), prec_lb, nselect, mask_w, max_prob_w.mean()
            else:
                max_idx_w = torch.cat((max_idx_w, max_idx_w.clone()))
                mask_w = max_idx_w == label
                ce = self.ce_ulb(output, label)

                nselect = sum(mask_w).detach()  # Count the num
                # Compute the loss for the selected data only
                unsup_loss = ce * mask_w
                #max_prob = max_prob_w * mask_w
                # Compute the training acc for these selected data only
                prec_ulb = accuracy(output.detach(), label * mask_w.detach(), topk=(1,))[0] * cosine.size()[0]
                return unsup_loss.mean(), prec_ulb, nselect, mask_w, max_prob_w.mean()
        else:
            cosine = F.linear(F.normalize(x), F.normalize(self.weight))

            pseudo_label_lb = torch.softmax(cosine, dim=-1)
            max_prob_lb, max_idx_lb = torch.max(pseudo_label_lb, dim=-1)

            sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            # max_prob, _ = torch.max(output, dim=-1)
            output = output * self.s
            loss_lb = self.ce_lb(output, label)
            prec_lb = accuracy(output.detach(), label.detach(), topk=(1,))[0]
            return loss_lb, prec_lb, max_idx_lb