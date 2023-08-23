import sys
import os

import numpy as np
import tqdm
import numpy
import time,copy
import gc
import soundfile
import faiss
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances
from tools import *
from copy import deepcopy
from loss import AAMLoss
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from encoder import ECAPA_TDNN

class trainer(nn.Module):
    def __init__(self, args, lr, n_cluster):
        super(trainer, self).__init__()
        self.Network = ECAPA_TDNN(num_class=n_cluster, C=1024).cuda()  # Speaker encoder
        # Classification layer
        #self.thresold = torch.ones((n_cluster))/n_cluster
        self.thresold = 1/n_cluster
        self.Loss = AAMLoss(n_class=n_cluster, margin=0.2, scale=30).cuda()
        self.Optim = torch.optim.Adam(list(self.Network.parameters())+list(self.Loss.parameters()), lr=lr, weight_decay = 2e-5)  # Adam, learning rate is fixed
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.Optim, T_max=30,eta_min=1e-5)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.Optim, T_0=20, T_mult=1, eta_min=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.Optim, step_size=args.test_interval, gamma=args.lr_decay)

    def train_network(self, args, epoch, loader_dict, gated, fix=True):
        self.train()
        loss, index, nselects_lb, nselects_ulb, top1_lb, top1_ulb, truetop1_ulb, num, tloss = 0, 0, 0, 0, 0, 0, 0, 1, 0
        time_start = time.time()
        lr = self.Optim.param_groups[0]['lr']
        yulb_true, yulb_pred, ylb_true, ylb_pred = [], [], [], []
        for (xw_lb, xs_lb, y_lb), (xw_ulb, xs_ulb, dict_y_ulb, true_y_ulb) in zip(cycle(loader_dict['train_lb']), loader_dict['train_ulb']):
            self.zero_grad()
            xw_lb, xs_lb, xw_ulb, xs_ulb = xw_lb.cuda(),xs_lb.cuda(), xw_ulb.cuda(), xs_ulb.cuda()
            inputs = torch.cat((xw_lb, xs_lb, xs_ulb))
            num_lb = xs_lb.shape[0]
            #label_lb = torch.LongTensor(y_lb)
            label_lb = torch.cat((torch.LongTensor(y_lb), torch.LongTensor(y_lb).clone()))

            dict_y_ulb = torch.LongTensor(dict_y_ulb)
            true_y_ulb = torch.LongTensor(true_y_ulb)
            dict_y_ulb = torch.cat((dict_y_ulb, dict_y_ulb.clone()))
            true_y_ulb = torch.cat((true_y_ulb, true_y_ulb.clone()))

            # input segment and the output speaker embedding
            embeddings_x = self.Network.forward(inputs, aug=True)

            embeddings_x_lb = embeddings_x[:2*num_lb]
            sup_loss, prec1_lb, label_lb_pred = self.Loss.forward(embeddings_x_lb, label_lb.cuda(), labelled=True)

            embeddings_xw_ulb = self.Network.forward(xw_ulb, aug=False)
            embeddings_xs_ulb = embeddings_x[2*num_lb:]
            embeddings_x_ulb = torch.cat((embeddings_xw_ulb, embeddings_xs_ulb))

            unsup_loss, prec1_ulb, nselect, index_mask, mean= self.Loss.forward(embeddings_x_ulb, dict_y_ulb.cuda(),
                                                                                thresold=self.thresold, labelled=False, gated=gated)
            if fix == False:
                self.thresold = 0.5 * self.thresold + 0.5 * mean.detach().cpu().numpy()

            nselect = nselect.detach()
            total_loss = sup_loss + args.ulb_loss_ratio * unsup_loss

            total_loss.backward()
            self.Optim.step()
            loss += total_loss.detach().cpu().numpy()

            ylb_true.extend(label_lb.cpu().tolist())
            ylb_pred.extend(label_lb_pred.cpu().tolist())
            yulb_true.extend(true_y_ulb[index_mask].cpu().tolist())
            yulb_pred.extend(dict_y_ulb[index_mask].cpu().tolist())
            #acc_selects += torch.sum(dictlabel_ulb[index_mask] == truelabel_ulb[index_mask])

            index += len(dict_y_ulb)
            truetop1_ulb += torch.sum(dict_y_ulb==true_y_ulb)
            nselects_lb += len(label_lb)
            top1_lb += prec1_lb
            nselects_ulb += nselect
            top1_ulb += prec1_ulb
            #show_acc_selects = acc_selects / nselects_ulb * 100
            time_used = time.time() - time_start
            sys.stderr.write(time.strftime("%H:%M")+"[%2d],Lr:%5f,lb:%.2f%%,ulb:%.2f%%(est%.1f mins),Loss:%.3f,la:%.2f%%,ta:%.2f%%,uls:%.2f%%,ulsa:%.2f%%,t:%.7f\r" %
                             (epoch, lr, 100 * (num / loader_dict['train_lb'].__len__()),
                              100 * (num / loader_dict['train_ulb'].__len__()),
                              time_used * loader_dict['train_ulb'].__len__() / num / 60,
                              loss / num, top1_lb / nselects_lb,
                              accuracy_score(ylb_true, ylb_pred)*100,
                              nselects_ulb / index * 100,
                              accuracy_score(yulb_true, yulb_pred)*100, self.thresold))
            num += 1
            sys.stderr.flush()
        self.scheduler.step()
        sys.stdout.write("\n")
        gc.collect()
        torch.cuda.empty_cache()
        slectedNMI = normalized_mutual_info_score(yulb_true, yulb_pred) * 100
        return loss / num, top1_lb / nselects_lb, top1_ulb / nselects_ulb, nselects_ulb / index * 100, accuracy_score(yulb_true,yulb_pred)*100, self.thresold, slectedNMI

    def train_network_lb(self, epoch, lb_trainLoader):
        self.train()
        time_start = time.time()
        index, top1_lb, loss = 0, 0, 0

        self.scheduler.step(epoch - 1)
        lr = self.Optim.param_groups[0]['lr']
        for num, (xw_lb, xs_lb, y_lb) in enumerate(lb_trainLoader, start=1):
            self.zero_grad()
            y_alllb = torch.cat((y_lb, torch.LongTensor(y_lb).clone())).cuda()
            x_lb = torch.cat((xw_lb, xs_lb))
            x_lb = x_lb.cuda()

            # input segment and the output speaker embedding
            embeddings = self.Network.forward(x_lb, aug=True)
            nloss, prec_lb, logits_lbmean = self.Loss.forward(embeddings, y_alllb, labelled=True)

            nloss.backward()
            self.Optim.step()

            index += len(y_alllb)
            top1_lb += prec_lb
            loss += nloss.detach().cpu().numpy()

            time_used = time.time() - time_start
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                             " [%2d], Lr: %5f,  labeleddata: %.2f%% (est %.1f mins), Loss: %.3f, ACC_label: %.2f%%\r" %
                             (epoch, lr, 100*(num/lb_trainLoader.__len__()),
                              time_used*lb_trainLoader.__len__()/num/60,
                              loss/num, top1_lb/index*len(y_alllb)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        torch.cuda.empty_cache()
        return loss/num, top1_lb/index*len(y_alllb)

    def cluster_network(self, args, clusterLoader, lb_trainLoader, n_cluster, score_file, epoch):
        self.eval()
        out_lb, filenames_lb, labels_lb = [], [], []
        for dataw_lb, datas_lb, filename_lb, label_lb in tqdm.tqdm(lb_trainLoader):
            with torch.no_grad():
                embedding_lb = self.Network.forward(dataw_lb.cuda(), aug=False)  # Get the embeddings
                embedding_lb = F.normalize(embedding_lb, p=2, dim=1)  # Normalization
                # Save the filname, labels, and the embedding into the list [labels is used to compute NMI]
                for i in range(len(filename_lb)):
                    filenames_lb.append(filename_lb[i])
                    labels_lb.append(label_lb[i].cpu().numpy())
                    out_lb.append(embedding_lb[i].detach().cpu().numpy())
        out_lb = numpy.array(out_lb)
        print(out_lb.shape)
        labels_lb = numpy.array(labels_lb)

        out_ulb, filenames_ulb, labels_ulb, logits_ulb = [], [], [],[]
        for data_ulb, filename_ulb, label_ulb in tqdm.tqdm(clusterLoader):
            with torch.no_grad():
                embedding_ulb = self.Network.forward(data_ulb[0].cuda(), aug=False)  # Get the embeddings
                embedding_ulb = F.normalize(embedding_ulb, p=2, dim=1)  # Normalization
                # Save the filname, labels, and the embedding into the list [labels is used to compute NMI]
                for i in range(len(filename_ulb)):
                    filenames_ulb.append(filename_ulb[i][0])
                    labels_ulb.append(label_ulb[i].cpu().numpy()[0])
                    out_ulb.append(embedding_ulb[i].detach().cpu().numpy())
        out_ulb = numpy.array(out_ulb)
        labels_ulb=numpy.array(labels_ulb)

        # Clustering using faiss https://github.com/facebookresearch/deepcluster
        #clus = faiss.Clustering(out_ulb.shape[1], n_cluster)
        preds_ulb=cskmeans(n_cluster, score_file, out_lb, labels_lb, out_ulb, labels_ulb, tolerance=0.05, max_iter=100)

        acc = np.sum(labels_ulb == preds_ulb)/out_ulb.shape[0] * 100
        print(labels_ulb.shape)
        print(accuracy_score(labels_ulb, preds_ulb)*100)

        #print('Clustering Accuracy of unlabeled data:',acc)

        del out_ulb, out_lb
        gc.collect()
        dic_label = defaultdict(list)  # Pseudo label dict

        for i in range(len(preds_ulb)):
            pred_ulblabel = preds_ulb[i]  # pseudo label
            filename = filenames_ulb[i]  # its filename
            dic_label[filename].append(pred_ulblabel)
            dic_label[filename].append(labels_ulb[i])
            #dic_label[filename] = pred_ulblabel, labels_ulb[i]  # save into the dic

        # Compute the NMI.
        NMI = normalized_mutual_info_score(labels_ulb, preds_ulb) * 100
        torch.cuda.empty_cache()
        return dic_label, NMI, acc

    def eval_network(self, val_list, val_path):
        self.eval()
        files, feats = [], {}
        for line in open(val_list).read().splitlines():
            data = line.split()
            files.append(data[1])
            files.append(data[2])
        setfiles = list(set(files))
        setfiles.sort()  # Read the list of wav files

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(val_path, file))
            feat = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            with torch.no_grad():
                ref_feat = self.Network.forward(feat,aug=False)
            # Extract features for each data, get the feature dict
            feats[file] = ref_feat

        scores, labels = [], []
        for line in open(val_list).read().splitlines():
            data = line.split()
            ref_feat = F.normalize(feats[data[1]], p=2, dim=1)  # feature 1
            com_feat = F.normalize(feats[data[2]], p=2, dim=1)  # feature 2
            score = numpy.mean(torch.matmul(ref_feat, com_feat.T).detach().cpu().numpy())  # Get the score
            scores.append(score)
            labels.append(int(data[0]))
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        return [EER, minDCF]

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        print("Model %s loaded!" % (path))
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    # print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" % (origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

def cskmeans(n_clusters, score_file, X, labels_lb, unlabeled_X, labels_ulb, tolerance=1e-2, max_iter=300):
    print('Initialization complete')
    print(max_iter)
    centroids = np.empty((n_clusters, X.shape[1]), dtype=X.dtype)
    for j in range(len(np.unique(labels_lb))):
        centroids[j] = np.mean(X[labels_lb == j], axis=0)
    cur_centers = centroids
    new_centers = deepcopy(centroids)

    X_all = np.vstack([X, unlabeled_X])
    # Main loop
    i=1
    while(1):
        time_start = time.time()
        print('Iteration', i, 'statr at', time.strftime("%m-%d %H:%M:%S"))

        index=faiss.IndexFlatIP(X.shape[1])
        index.add(cur_centers)

        preds_ulb = [int(i[0]) for i in index.search(unlabeled_X, 1)[1]]
        y_label = np.hstack([labels_lb, preds_ulb])

        # Second step: update each centroids
        for j in range(len(np.unique(labels_lb))):
            new_centers[j] = X_all[y_label == j].mean(axis=0)
        # Check if KMeans converges
        #difference = np.linalg.norm(new_centers - cur_centers, ord='fro')
        diff = np.abs(new_centers - cur_centers)
        max_diff=diff.max()
        #diff=np.linalg.norm(new_centers-cur_centers, ord=2, axis=1)
        #max_diff = max(diff)
        NMI = normalized_mutual_info_score(labels_ulb, preds_ulb) * 100
        acc=accuracy_score(labels_ulb, preds_ulb)
        time_used = time.time() - time_start
        score_file.write('Iteration [%2d]:difference:%.7f, NMI:%.7f, Acc:%.7f%% time:%.1f mins\n' % (i, max_diff, NMI, acc*100, time_used / 60))
        print('Iteration [%2d]:difference:%.7f, NMI:%.7f, Acc:%.7f%% time:%.1f mins\n' % (i, max_diff, NMI, acc*100, time_used / 60))
        cur_centers = deepcopy(new_centers)

        if max_diff<tolerance or i == max_iter:
            print('Converged at iteration {}.\n'.format(i))
            return preds_ulb
        i += 1