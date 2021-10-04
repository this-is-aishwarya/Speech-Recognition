#prepare_dataset_gfcc
import os
import json
import scipy.io.wavfile
from scipy.io.wavfile import read as read_wav
import spafe
import librosa
from spafe.features.lfcc import lfcc

DATASET_PATH = "audio"
JSON_PATH = "data_lfcc.json"
SAMPLES_TO_CONSIDER = 16000

def preprocess_dataset(dataset_path, json_path, num_ceps = 13, low_freq = 0, high_freq = 2000, nfilts = 24, nfft = 512, dct_type = 2, use_energy = False,lifter = 5,normalize = False):
    """Extracts GFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save BFCCs
    :param num_ceps (int): Number of coefficients to extract
    :return:
    """

    # dictionary where we'll store mapping, labels, BFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "LFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:

            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            # process all audio files in sub-dir and store MFCCs
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # read wav 
                fs, sig = scipy.io.wavfile.read(file_path)

                if len(sig) >= SAMPLES_TO_CONSIDER:

                    sig = sig[:SAMPLES_TO_CONSIDER]
                    
                    #BFCCs  = bfcc(sig, 13)
                    # compute features
                    LFCCs = lfcc(sig=sig, fs=fs,num_ceps=num_ceps, nfilts=nfilts, nfft=nfft, low_freq=low_freq, high_freq=high_freq, dct_type=dct_type, use_energy=use_energy, lifter=lifter, normalize=normalize)


                    # store data for analysed track
                    data["LFCCs"].append(LFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)