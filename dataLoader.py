import glob
import numpy
import os
import random
import soundfile
import torch
import wave
from scipy import signal
from tools import *
from collections import defaultdict

def get_dataloader(args, lb_data, lb_label, ulb_data, ulb_truelabel, ulb_length, dic_label=None, cluster_only=False):
    loader_dict = {}
    clusterLoader = cluster_loader(ulb_data, ulb_truelabel, ulb_length)
    clusterLoader = torch.utils.data.DataLoader(clusterLoader, batch_size=1, shuffle=True, num_workers=args.n_cpu, drop_last=False)
    if cluster_only == True:
        lb_trainLoader = train_loader(args, lb_data, lb_label, cluster_only=True)
        lb_trainLoader = torch.utils.data.DataLoader(lb_trainLoader, batch_size=100, shuffle=True, num_workers=args.n_cpu, drop_last=False)
        return clusterLoader, lb_trainLoader
    else:
        #dic_label_lb = convert(lb_data, lb_label)
        trainLoader = train_loader(args, ulb_data, dic_label, cluster_only=False, labelled=False)
        loader_dict['train_ulb'] = torch.utils.data.DataLoader(trainLoader, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)
        lb_trainLoader = train_loader(args, lb_data, lb_label, cluster_only=False, labelled=True)
        loader_dict['train_lb'] = torch.utils.data.DataLoader(lb_trainLoader, batch_size=args.batch_size*args.uratio, shuffle=True,num_workers=args.n_cpu, drop_last=True)
        return clusterLoader, loader_dict

class train_loader(object):
    def __init__(self, args, data_list, data_label, cluster_only, labelled=True):
        self.train_path = args.train_path
        self.labelled=labelled
        self.cluster_only=cluster_only
        self.max_frames = args.max_frames * 160 + 240
        # Load and configure augmentation files
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise': [0, 15],'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(args.musan_path, '*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files = glob.glob(os.path.join(args.rir_path, '*/*/*.wav'))
        self.data_list = data_list
        self.label = data_label

    def __getitem__(self, index):
        file = self.data_list[index]
        audio, sr = soundfile.read(file)
        if audio.shape[0] <= self.max_frames:
            shortage = self.max_frames - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - self.max_frames))
        audio = audio[start_frame:start_frame + self.max_frames]
        audio = numpy.stack([audio], axis=0)

        # Data Augmentation
        if self.cluster_only == True:
            augtype = random.randint(0, 5)
            aug_audio = self.augmentaudio(audio, augtype)
            label = self.label[index]
            return torch.FloatTensor(audio[0]), torch.FloatTensor(aug_audio[0]), file, label
        else:
            if self.labelled==True:
                augtype = random.randint(1, 5)
                aug_audio = self.augmentaudio(audio, augtype)
                lb_label = self.label[index]
                return torch.FloatTensor(audio[0]), torch.FloatTensor(aug_audio[0]), lb_label
            else:
                augtype = random.randint(1, 5)
                aug_audio = self.augmentaudio(audio, augtype)
                dictlabel = self.label[file][0]
                dicttruelabel = self.label[file][1]
                return torch.FloatTensor(audio[0]), torch.FloatTensor(aug_audio[0]), dictlabel, dicttruelabel

    def __len__(self):
        return len(self.data_list)

    def augmentaudio(self,audio, augtype):
        #augtype = random.randint(1, 5)
        if augtype == 0: # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2:  # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return audio

    def add_rev(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        return signal.convolve(audio, rir, mode='full')[:, :self.max_frames]

    def add_noise(self, audio, noisecat):
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            if noiseaudio.shape[0] <= self.max_frames:
                noiseaudio = numpy.pad(noiseaudio, (0, self.max_frames - noiseaudio.shape[0]), 'wrap')
            start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - self.max_frames))
            noiseaudio = noiseaudio[start_frame:start_frame + self.max_frames]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio

class cluster_loader(object):
    def __init__(self, data_list, data_truelabel, data_length):
        self.data_list, self.data_length, self.data_label = data_list, data_length, data_truelabel

        # sort the training set by the length of the audios, audio with similar length are saved togethor.
        inds = numpy.array(self.data_length).argsort()
        self.data_list, self.data_length, self.data_label = numpy.array(self.data_list)[inds], \
            numpy.array(self.data_length)[inds], \
            numpy.array(self.data_label)[inds]
        self.minibatch = []
        start = 0
        while True:  # Genearte each minibatch, audio with similar length are saved togethor.
            frame_length = self.data_length[start]
            minibatch_size = max(1, int(1600 // frame_length))
            end = min(len(self.data_list), start + minibatch_size)
            self.minibatch.append([self.data_list[start:end], frame_length, self.data_label[start:end]])
            if end == len(self.data_list):
                break
            start = end

    def __getitem__(self, index):
        # Get one minibatch
        data_lists, frame_length, data_labels = self.minibatch[index]
        filenames, labels, segments = [], [], []
        for num in range(len(data_lists)):
            filename = data_lists[num]  # Read filename
            label = data_labels[num]  # Read GT label
            audio, sr = soundfile.read(os.path.join(filename))
            if len(audio) < int(frame_length * sr):
                shortage = int(frame_length * sr) - len(audio) + 1
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            # Get clean utterance, better for clustering
            audio = numpy.array(audio[:int(frame_length * sr)])
            segments.append(audio)
            filenames.append(filename)
            labels.append(label)
        segments = torch.FloatTensor(numpy.array(segments))
        return segments, filenames, labels

    def __len__(self):
        return len(self.minibatch)
