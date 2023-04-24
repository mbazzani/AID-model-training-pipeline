#!/usr/bin/env python3
import os
import pandas as pd

class BirdSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transform=None, target_transform=None):
        labels_df = pd.read_csv(annotations_file)
        self.audio_labels = labels_df["NEW NAME", "MANUAL ID"]
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise StopIteration

        clip_path = os.path.join(self.audio_dir, self.audio_labels.iloc[idx, 0])
        clip = load_clip(clip_path)
        
        if clip is None:
            #TODO Add error handling/logging
            del self.audio_labels[idx]
            return self.__getitem__(index)
        else:
            label = self.audio_labels.iloc[idx, 1]
            if self.transform:
                clip = self.transform(clip)
            if self.target_transform:
                label = self.target_transform(label)
            return clip, label
