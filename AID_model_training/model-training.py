#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
import torchvision
from pathlib import Path
from matplotlib import pyplot as plt
import WTS_chunking
import librosa
import soundfile
import scipy.signal
import logging
import re
import os

# Paths having spaces switched to underscores
CHUNK_LENGTH = 3
LABELED_CSV_PATH = "../datasets/cosmos/cosmos_labeled.csv"
DATA_PATH = "../datasets/cosmos/cosmos_data/"
VERBOSE = True

def main():
    labeled_data_df = pd.read_csv(labeled_csv_path)
    print(labeled_data_df["IN FILE"])
    labeled_data_df["IN FILE"] = labeled_data_df["IN FILE"].str.replace("_"," ")
    print(labeled_data_df["IN FILE"])
    chunked_df = generate_chunked_df(labeled_data_df)
    chunked_df["FOLDER"] = "cosmos_chunked/"
    #Switch to efficientnet4
    #model = torchvision.models.resnet50(weights=None)
    split_clips_into_chunks(labeled_data_df, chunked_df)

def check_verbose(message):
    if VERBOSE: print(message)

#Function to convert clip to trainig data given path
def loader(path):
    signal = librosa.load(path)
    return signal

#Function to split training data into chunks,
#given a dataframe representing the chunks
def split_clips_into_chunks(original_clips_df, chunked_df):
    folder = chunked_df["FOLDER"][0]
    chunked_path = DATA_PATH + folder
    if not os.path.exists(chunked_path):
        os.mkdir(chunked_path)
    # make directory for clips to go in
    for clip_name in original_clips_df["IN FILE"].unique():
        current_chunks_df = chunked_df.loc[chunked_df["IN FILE"] == clip_name]
        loaded_clip = load_clip(DATA_PATH + str(clip_name))
        if loaded_clip is not None:
            full_clip_signal, sample_rate = loaded_clip
        else:
            continue
        for i, row in current_chunks_df.iterrows():
            chunk_start = int(row["OFFSET"]*sample_rate)
            chunk_end = int(chunk_start + CHUNK_LENGTH*sample_rate)
            chunk_signal = full_clip_signal[chunk_start:chunk_end]# Something with full signal, offset, etc.
            soundfile.write(chunked_path + row["NEW NAME"], chunk_signal, sample_rate)


# Takes the and generates a dataframe of path names
# for chunked clips from it
def generate_chunked_df(original_clips_df):
    chunked_df = WTS_chunking.annotation_chunker_no_duplicates(original_clips_df,
                                                               CHUNK_LENGTH)
    #For each chunk, append the chunk's position in the clip to the clip name
    chunked_df = update_names(chunked_df)
    return chunked_df

def update_names(chunked_df):
    chunked_df[['FILE NAMES', 'FILE EXTENSIONS']] = chunked_df['IN FILE'].str.split('.', 1, expand=True)
    chunked_df["NEW NAME"] = chunked_df["FILE NAMES"]\
        + chunked_df["OFFSET"].astype('int').astype('string')\
        + "." + chunked_df["FILE EXTENSIONS"]
    chunked_df.drop(["FILE NAMES", "FILE EXTENSIONS"], axis='columns')
    return chunked_df


def load_clip(clip_path):
    try:

        signal, sample_rate = librosa.load(clip_path, sr=None, mono=True)
        # Librosa uses range of 0-1, model needs range of -32768 to 32768
        signal = signal * 32768
    except BaseException:
        check_verbose("Failure in loading" + clip_path)
        return
    # Downsample the audio if the sample rate > 44.1 kHz
    # Force everything into the human hearing range.
    try:
        if sample_rate > 44100:
            rate_ratio = 44100 / sample_rate
            signal = scipy.signal.resample(signal, int(len(signal) * rate_ratio))
            sample_rate = 44100
    except BaseException:
        check_verbose("Failure in downsampling" + clip_path)
        return

    # Converting to Mono if Necessary
    if len(signal.shape) == 2:
        # averaging the two channels together
        signal = signal.sum(axis=1) / 2

    return signal, sample_rate

if __name__=="__main__":
    main()
