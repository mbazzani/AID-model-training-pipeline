#!/usr/bin/env python3
import os
import soundfile
import librosa
import scipy.signal
import pandas as pd
import numpy as np
import config as cfg
import WTS_chunking


def create_chunked_data():
    original_labeled_df = pd.read_csv(cfg.ORIGINAL_DATA_DIR / cfg.LABELS_FILE)

    # For some reason pandas replaces spaces with underscores when reading the
    # csv, this line just undoes that so it is consistent with the original
    original_labeled_df["IN FILE"] = original_labeled_df["IN FILE"].str.replace(
        "_", " "
    )

    chunked_df = generate_chunked_df(original_labeled_df)
    chunked_df["FOLDER"] = cfg.CHUNKED_DATA_DIR

    split_clips_into_chunks(original_labeled_df, chunked_df)


def check_verbose(message, verbose):
    if verbose:
        print(message)


# Function to split training data into chunks,
# given a dataframe representing the chunks
def split_clips_into_chunks(original_clips_df, chunked_df):
    # make directory for clips to go in
    if not os.path.exists(cfg.CHUNKED_DATA_DIR):
        os.mkdir(cfg.CHUNKED_DATA_DIR)

    if not os.path.exists(cfg.CHUNKED_DATA_DIR / cfg.LABELS_FILE):
        chunked_df.to_csv(cfg.CHUNKED_DATA_DIR / cfg.LABELS_FILE)

    for clip_name in original_clips_df["IN FILE"].unique():
        current_chunks_df = chunked_df.loc[chunked_df["IN FILE"] == clip_name]
        loaded_clip = load_clip(cfg.ORIGINAL_DATA_DIR / str(clip_name))

        if loaded_clip is not None:
            full_clip_signal, _ = loaded_clip

        current_chunks_df.apply(lambda row: write_chunks(
            row, full_clip_signal), axis=1)


def write_chunks(row, full_clip_signal):
    chunk_start = int(row["OFFSET"] * cfg.SAMPLE_RATE)
    chunk_end = int(chunk_start + cfg.CHUNK_LENGTH * cfg.SAMPLE_RATE)
    chunk_signal = full_clip_signal[chunk_start:chunk_end]
    soundfile.write(
        cfg.CHUNKED_DATA_DIR / row["NEW NAME"], chunk_signal, cfg.SAMPLE_RATE
    )


# Takes the and generates a dataframe of path names
# for chunked clips from it
def generate_chunked_df(original_clips_df):
    chunked_df = WTS_chunking.annotation_chunker_no_duplicates(
        original_clips_df, cfg.CHUNK_LENGTH
    )
    # For each chunk, append the chunk's position in the clip to the clip name
    chunked_df = update_names(chunked_df)
    return chunked_df


# Give chunks new file names based on the file
# they came from and their position in it
def update_names(chunked_df):
    # Doesn't work with pandas 2.0 for some reason? Needs version <=1.5.3
    chunked_df[["FILE NAMES", "FILE EXTENSIONS"]] = chunked_df["IN FILE"].str.rsplit(
        pat=".", n=1, expand=True
    )

    chunked_df["NEW NAME"] = (
        chunked_df["FILE NAMES"]
        + chunked_df["OFFSET"].astype("int").astype("string")
        + "."
        + chunked_df["FILE EXTENSIONS"]
    )

    chunked_df.drop(["FILE NAMES", "FILE EXTENSIONS"], axis="columns")
    return chunked_df


def load_clip(clip_path, verbose=False):
    try:
        signal, sample_rate = librosa.load(
            str(clip_path), sr=None, mono=True
        )  # Librosa uses range of 0-1, model needs range of -32768 to 32768
        signal = signal * 32768
    except BaseException:
        check_verbose("Failure in loading: " + str(clip_path), verbose)
        return

    # Downsample the audio if the sample rate > 44.1 kHz
    # Force everything into the human hearing range.
    try:
        if sample_rate > 44100:
            rate_ratio = 44100 / sample_rate
            signal = scipy.signal.resample(
                signal, int(len(signal) * rate_ratio))
            sample_rate = 44100
    except BaseException:
        check_verbose("Failure in downsampling" / clip_path, verbose)
        return

    # Converting to Mono if Necessary
    assert isinstance(signal, np.ndarray)
    if len(signal.shape) == 2:
        # averaging the two channels together
        signal = signal.sum(axis=1) / 2

    # This is hacky, need to figure out why some clips are too 
    # short in the first place
    if len(signal) < cfg.SAMPLE_RATE * cfg.CHUNK_LENGTH:
        return None

    return signal, sample_rate
