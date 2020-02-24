import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LayerNormalization

from os import listdir
from os.path import isfile, join, isdir
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot  as plt

class Config:
    TEST_PATHS = [
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test019",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test024",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test014",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test030",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test027"
        ]
    MODEL_PATH = "model_lstm.hdf5"
    RESULTS_FILE = "test_results.csv"

def get_model():
    tf.keras.backend.set_floatx('float32')
    return load_model(
        Config.MODEL_PATH, 
        custom_objects = {'LayerNormalization': LayerNormalization})

def get_test(test_idx: int):
    sz = 200
    test = np.zeros(shape=(sz, 256, 256, 1))
    cnt = 0
    for f in sorted(listdir(Config.TEST_PATHS[test_idx])):
        if str(join(Config.TEST_PATHS[test_idx], f))[-3:] == "tif":
            #img = cv2.imread(join(Config.TEST_PATHS[0], f))
            #img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
            img = Image.open(join(Config.TEST_PATHS[test_idx], f)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[cnt, :, :, 0] = img
            cnt = cnt + 1
    return test

def sequence_from_data(data):
    sz = data.shape[0] - 10
    sequences = np.zeros((sz, 10, 256, 256, 1))
    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = data[i + j, :, :, :]
        sequences[i] = clip
    return sequences

def evaluate():
    model = get_model()
    out_dict = []
    for idx in len(Config.TEST_PATHS):
        out = {}
        out["Path"] = Config.TEST_PATHS[idx]
        test = get_test(idx)
        sequences = sequence_from_data(test)
        reconstructed_sequences = model.predict(sequences, batch_size = 4)
        sequences_reconstruction_cost = np.array(
            [np.linalg.norm(
                np.subtract(sequences[i],reconstructed_sequences[i])) 
                for i in range(0,sz)])
        sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
        sr = 1.0 - sa
        out["Sequence Cost"] = sequences_reconstruction_cost
        out["Score"] = sr
        out_dict.append(out)

    with open(Config.RESULTS_FILE, 'w') as f:
        json.dump(out_dict, f)
        

evaluate()



