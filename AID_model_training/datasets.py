#!/usr/bin/env python3
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import training_data_processing as data_processing
from torchaudio import transforms as audtr
import config as cfg


class BirdSoundDataset(Dataset):
    def __init__(self, labels_file, audio_dir, transform=None, target_transform=None):
        labels_df = pd.read_csv(labels_file)
        self.audio_labels = labels_df[["NEW NAME", "MANUAL ID"]]
        self.audio_labels["MANUAL ID"] = self.audio_labels["MANUAL ID"].map(cfg.LABEL_MAP)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #print(len(self.audio_labels))
        return len(self.audio_labels)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration

        clip_path = os.path.join(self.audio_dir, self.audio_labels.iloc[index, 0])
        clip = data_processing.load_clip(clip_path)
        if clip is None:
            return None
        clip,_ = clip
        mel_spectrogram = audtr.MelSpectrogram(sample_rate=cfg.SAMPLE_RATE, 
                                    n_mels=cfg.NUM_MEL_FILTERBANKS, 
                                    n_fft=cfg.FFT_SIZE)
        # Should be a tensor with size batch_size, num_samples
        clip = mel_spectrogram(torch.tensor(clip))
        clip = torch.stack([clip,clip,clip])

        label = self.audio_labels.iloc[index, 1]

        if self.transform:
            clip = self.transform
        if self.target_transform:
            label = self.target_transform(label)
        return clip, label

def get_datasets(labels_file, audio_dir, device):
    data = BirdSoundDataset(labels_file, audio_dir)
    valid_indices = [i for i in range(len(data)) if data[i] is not None]
    valid_dataset = torch.utils.data.Subset(data, valid_indices)
    train_data, val_data = torch.utils.data.random_split(valid_dataset, [0.8, 0.2])
    return train_data, val_data

