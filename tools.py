import warnings
import torch
import os
import math
import numpy
import yaml
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
from operator import itemgetter
import argparse
from torch.optim import lr_scheduler

class WarmupCosineLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_max, lr_min=1e-4, warm_up=0, epoch=10, start_ratio=0.1):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_min: (float), minimum learning rate
            - lr_max: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - epoch: (int), maximum epoch or iteration
            - start_ratio: (float), to control epoch 0 lr, if ratio=0, then epoch 0 lr is lr_min

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.warm_up = warm_up
        self.epoch = epoch
        self.start_ratio = start_ratio
        self.cur = 0    # current epoch or iteration

        super().__init__(optimizer, -1)

    def get_lr(self):
        if (self.warm_up == 0) & (self.cur == 0):
            lr = self.lr_max
        elif (self.warm_up != 0) & (self.cur <= self.warm_up):
            if self.cur == 0:
                lr = self.lr_min + (self.lr_max - self.lr_min) * \
                    (self.cur + self.start_ratio) / self.warm_up
            else:
                lr = self.lr_min + (self.lr_max - self.lr_min) * \
                    (self.cur) / self.warm_up
                # print(f'{self.cur} -> {lr}')
        else:
            # this works fine
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 *\
                (math.cos((self.cur - self.warm_up) /
                          (self.epoch - self.warm_up) * math.pi) + 1)

        self.cur += 1

        return [lr for base_lr in self.base_lrs]

def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_data(train_list, train_path):
    # Load data & labels
    data_list, data_label, data_length = [], [], []
    lines = open(train_list).read().splitlines()
    dictkeys = list(set([x.split()[0] for x in lines]))
    dictkeys.sort()
    dictkeys = {key: ii for ii, key in enumerate(dictkeys)}
    for index, line in enumerate(lines):
        data = line.split()
        file_name = os.path.join(train_path, data[1])
        file_length = float(data[-1])
        speaker_label = dictkeys[data[0]]
        data_list.append(file_name)  # Filename
        data_length.append(file_length)  # Filelength
        data_label.append(speaker_label)  # GT Speaker label
    return data_list, data_label, data_length


def split_ssl_data(data, target, data_length, num_labels, num_classes, include_lb_to_ulb=False):
    """
    data & target is splitted into labeled and unlabeld data.

    Args
            index: If np.array of index is given, select the data[index], target[index] as labeled samples.
            include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target, data_length = np.array(data), np.array(target), np.array(data_length)
    lb_data, lbs, lb_length, lb_idx, = sample_labeled_data(data, target, data_length, num_labels, num_classes)
    # unlabeled_data index of data
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx))))
    if include_lb_to_ulb:
        return lb_data, lbs, lb_length, data, target, data_length
    else:
        return lb_data, lbs, lb_length, data[ulb_idx], target[ulb_idx], data_length[ulb_idx]

def sample_labeled_data(data, target, data_length, num_labels, num_classes):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    lb_samples_per_class=int(num_labels/num_classes)
    print(lb_samples_per_class)
    lb_data, lbs, lb_idx, lb_length = [], [], [], []
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        if len(idx) >= lb_samples_per_class:
            idx = np.random.choice(idx, lb_samples_per_class, False)
        else:
            idx = np.random.choice(idx, lb_samples_per_class, True)
        lb_idx.extend(idx)
        lb_data.extend(data[idx])
        lbs.extend(target[idx])
        lb_length.extend(data_length[idx])

    return np.array(lb_data), np.array(lbs), np.array(lb_length), np.array(lb_idx)


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


# A helper function to print the text and write the log
def print_write(type, text, score_file):
    if type == 'T':  # Classification training without LGL (Baseline)
        epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI = text
        print("%d epoch, LOSS %f, lb_ACC %.2f%%, ulb_ACC %.2f%%, nselects %.2f%%, selects_trueacc %.2f%%, thresold: %.8f, slectedNMI: %.2f\n" % (epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold, slectedNMI))
        score_file.write("[T], %d epoch, LOSS %f, lb_ACC %.2f%%, ulb_ACC %.2f%%, nselects %.2f%%, selects_trueacc %.2f%%, thresold: %.8f, slectedNMI: %.2f\n" % (epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI))
    elif type == 'L':  # Classification training with LGL (Propose)
        epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI = text
        print("%d epoch, LOSS %f, lb_ACC %.2f%%, ulb_ACC %.2f%%, nselects %.2f%%, selects_trueacc %.2f%%, thresold: %.8f, slectedNMI: %.2f\n" % (epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI))
        score_file.write("[L], %d epoch, LOSS %f, lb_ACC %.2f%%, ulb_ACC %.2f%%, nselects %.2f%%, selects_trueacc %.2f%%, thresold: %.8f, slectedNMI: %.2f\n" % (epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI))
    elif type == 'V':  # Classification training with LGL (Propose)
        epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI = text
        print("%d epoch, LOSS %f, lb_ACC %.2f%%, ulb_ACC %.2f%%, nselects %.2f%%, selects_trueacc %.2f%%, thresold: %.8f, slectedNMI: %.2f\n" % (epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI))
        score_file.write("[V], %d epoch, LOSS %f, lb_ACC %.2f%%, ulb_ACC %.2f%%, nselects %.2f%%, selects_trueacc %.2f%%, thresold: %.8f, slectedNMI: %.2f\n" % (epoch, loss, lb_acc, ulb_acc, nselects, se_trueacc, thresold,slectedNMI))
    elif type == 'C':  # Clustering step
        epoch, NMI, ulbacc = text
        print("%d epoch, NMI %.2f, acc %.2f%%\n" % (epoch, NMI, ulbacc))
        score_file.write("[C], %d epoch, NMI %.2f, acc %.2f%%\n" % (epoch, NMI, ulbacc))
    elif type == 'E':  # Evaluation step
        epoch, EER, minDCF, bestEER = text
        print("EER %2.2f%%, minDCF %2.3f%%, bestEER %2.2f%%\n" % (EER, minDCF, bestEER))
        score_file.write("[E], %d epoch, EER %2.2f%%, minDCF %2.3f%%, bestEER %2.2f%%\n" % (epoch, EER, minDCF, bestEER))
    score_file.flush()

def check_clustering(score_path, LGL):  # Read the score.txt file, judge the next stage
    lines = open(score_path).read().splitlines()

    if LGL == True:  # For LGL, the order is
        # Iteration 1: (C-T-T...-T-L-L...-L-V-V...-V)
        # Iteration 2: (C-T-T...-T-L-L...-L--V-V...-V)
        # ...
        EERs_T, epochs_T, EERs_L, epochs_L, EERs_V, epochs_V, EERs, epochs = [], [], [], [], [], [], [], []
        iteration = 0
        train_type = 'T'
        is_cluster = False
        for line in lines:
            if line.split(',')[0] == '[C]':  # Clear all results after clustering
                EERs_T, epochs_T, EERs_L, epochs_L, EERs_V, epochs_V, EERs, epochs = [], [], [], [], [], [], [], []
                train_type = 'T'
                is_cluster = True
                iteration += 1
            # Save the evaluation result in this iteration
            elif line.split(',')[0] == '[E]':
                epoch = int(line.split(',')[1].split()[0])
                EER = float(line.split(',')[-3].split()[-1][:-1])
                if train_type == 'T':
                    epochs_T.append(epoch)
                    EERs_T.append(EER)  # Result in [T]
                    epochs.append(epoch)
                    EERs.append(EER)  # Result in [T]
                    is_cluster = False
                elif train_type == 'L':
                    epochs_L.append(epoch)
                    EERs_L.append(EER)  # Result in [L]
                    epochs.append(epoch)
                    EERs.append(EER)  # Result in [T]
                    is_cluster = False
                elif train_type == 'V':
                    epochs_V.append(epoch)
                    EERs_V.append(EER)  # Result in [V]
                    epochs.append(epoch)
                    EERs.append(EER)  # Result in [T]
                    is_cluster = False
            elif line.split(',')[0] == '[T]':  # If the stage is [T], record it
                train_type = 'T'
                is_cluster = False
            elif line.split(',')[0] == '[L]':  # If the stage is [L], record it
                train_type = 'L'
                is_cluster = False
            elif line.split(',')[0] == '[V]':  # If the stage is [V], record it
                train_type = 'V'
                is_cluster = False

        # The stage is [T], so need to judge the next step is keeping [T]? Or do LGL for [L] ?
        if train_type == 'T':
            if len(EERs_T) < 4:  # Too short training epoch, keep training
                if is_cluster==True:
                    return 'T', None, None, iteration, True
                else:
                    return 'T', None, None, iteration, False
            else:
                # Get the best training result already, go LGL
                if EERs_T[-1] > min(EERs_T) and EERs_T[-2] > min(EERs_T) and EERs_T[-3] > min(EERs_T):
                    best_epoch = epochs_T[EERs_T.index(min(EERs_T))]
                    next_epoch = epochs_T[-1]
                    return 'L', best_epoch, next_epoch, iteration, True
                else:
                    return 'T', None, None, iteration, False  # EER can still drop, keep training

        elif train_type == 'L':
            if len(EERs_L) < 4:  # Too short training epoch, keep LGL training
                return 'L', None, None, iteration, False
            else:
                # Get the best LGL result already, go clustering
                if EERs_L[-1] > min(EERs_L) and EERs_L[-2] > min(EERs_L) and EERs_L[-3] > min(EERs_L):
                    best_epoch = epochs[EERs.index(min(EERs))]
                    next_epoch = epochs_L[-1]
                    # Clustering based on the best epoch is more robust
                    return 'V', best_epoch, next_epoch, iteration, True
                else:
                    return 'L', None, None, iteration, False  # EER can still drop, keep training

        elif train_type == 'V':
            if len(EERs_V) < 4:  # Too short training epoch, keep LGL training
                return 'V', None, None, iteration, False
            else:
                # Get the best LGL result already, go clustering
                if EERs_V[-1] > min(EERs_V) and EERs_V[-2] > min(EERs_V) and EERs_V[-3] > min(EERs_V):
                    best_epoch = epochs[EERs.index(min(EERs))]
                    next_epoch = epochs_V[-1]
                    # Clustering based on the best epoch is more robust
                    return 'C', best_epoch, next_epoch, iteration, True
                else:
                    return 'V', None, None, iteration, False  # EER can still drop, keep training

    else:  # Baseline approach without LGL
        EERs_T, epochs_T = [], []
        iteration = 0
        is_cluster=False
        for line in lines:
            if line.split(',')[0] == '[C]':  # Clear all results after clustering
                EERs_T, epochs_T = [], []
                iteration += 1
                is_cluster = True
            elif line.split(',')[0] == '[E]':  # Save the evaluation result
                epoch = int(line.split(',')[1].split()[0])
                EER = float(line.split(',')[-3].split()[-1][:-1])
                epochs_T.append(epoch)
                EERs_T.append(EER)
                is_cluster = False

        if len(EERs_T) < 4:  # Too short training epoch, keep training
            if is_cluster == True:
                return 'T', None, None, iteration, is_cluster
            else:
                return 'T', None, None, iteration, False
        else:
            # Get the best training result, go clustering
            if EERs_T[-1] > min(EERs_T) and EERs_T[-2] > min(EERs_T) and EERs_T[-3] > min(EERs_T):
                best_epoch = epochs_T[EERs_T.index(min(EERs_T))]
                next_epoch = epochs_T[-1]
                return 'C', best_epoch, next_epoch, iteration, True
            else:
                return 'T', None, None, iteration, False  # EER can still drop, keep training

def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        # numpy.where(fpr<=tfa)[0][-1]
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return tunedThreshold, eer, fpr, fnr

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.


def ComputeErrorRates(scores, labels):

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
