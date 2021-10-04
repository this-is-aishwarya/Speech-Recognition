import scipy.io.wavfile
from spafe.features.gfcc import gfcc
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data_gfcc.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec. of audio

def preprocess_dataset(dataset_path, json_path, num_ceps = 13, low_freq = 0, high_freq = 2000, nfilts = 26, nfft = 512, dct_type = 2, use_energy = False, lifter = 5, normalize = 1, scale="constant"):
    """Extracts GFCCs from music dataset and saves them into a json file.

    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save GFCCs
    :param num_mfcc (int): Number of coefficients to extract
    :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :param hop_length (int): Sliding window for FFT. Measured in # of samples
    :return:
    """
    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
        "mapping": [],
        "labels": [],
        "GFCCs": [],
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

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # read the wav file
                fs, sig = scipy.io.wavfile.read(file_path)

                # if len(signal) >= SAMPLES_TO_CONSIDER:

                #     # ensure consistency of the length of the signal
                #     signal = signal[:SAMPLES_TO_CONSIDER]

                # compute features
                # GFCCs = gfcc(sig=signal, fs=16000, num_ceps=num_ceps, nfilts=nfilts, nfft=nfft, low_freq=low_freq, high_freq=high_freq, dct_type=dct_type, use_energy=use_energy, lifter=lifter, normalize=normalize, scale=scale)
                GFCCs  = gfcc(sig, num_ceps=13)

                # store data for analysed track
                data["GFCCs"].append(GFCCs.T.tolist())
                data["labels"].append(i-1)
                data["files"].append(file_path)
                print("{}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)