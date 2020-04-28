#import tensorflow as tf
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import LayerNormalization

import sys
from os import listdir
from os.path import isfile, join, isdir, exists
import cv2
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot  as plt
import csv

class TestPaths:
    PATH = "warehouse_dataset/DSCWillmer/Test"

class Config:
    RESULTS_FILE = "test_results_all.csv"
    BASE_MODEL_PATH = "model_lstm.hdf5"
    UCSD_MODEL_PATH = "UCSD_multi_model_lstm.hdf5"
    MULTI_MODEL_PATH = "UCSD+Avenue_model_lstm.hdf5"
    MODEL_PATHS = ["willmer_model.hdf5"]
    IMAGE_SIZE = 256

imsize = Config.IMAGE_SIZE

def get_model(model):
    tf.keras.backend.set_floatx('float32')
    return load_model(
        model,
        custom_objects = {'LayerNormalization': LayerNormalization})

def get_test(test_path: str):
    sz = len([name for name in listdir(test_path) if name.endswith('tif') or name.endswith('png') or name.endswith('jpeg')])
    test = np.zeros(shape=(sz, imsize, imsize, 1))
    cnt = 0
    for f in sorted(listdir(test_path)):
        if str(join(test_path, f)).endswith(("tif")):
            #img = cv2.imread(join(Config.TEST_PATHS[0], f))
            #img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
            img = Image.open(join(test_path, f)).resize((imsize, imsize))
        elif  str(join(test_path, f)).endswith(("png","jpeg")):
            img = Image.open(join(test_path, f)).resize((imsize, imsize)).convert('L')
        else: continue
        img = np.array(img, dtype=np.float32) / float(256)
        test[cnt, :, :, 0] = img
        cnt = cnt + 1
    return test

def sequence_from_data(data):
    sz = data.shape[0] - 10
    sequences = np.zeros((sz, 10, imsize, imsize, 1))
    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        clip = np.zeros((10, imsize, imsize, 1))
        for j in range(0, 10):
            clip[j] = data[i + j, :, :, :]
        sequences[i] = clip
    return sequences

def evaluate():
    """ FOR HOLDING THE WHOLE CSV IN ONE PLACE
    fieldnames = ["Model Path", "Test Path", "Total Frames", "Sequence Cost", "Score"]
    with open(Config.RESULTS_FILE, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    """
    results_folder = "Results/DSCWillmer/"
    for model_str in Config.MODEL_PATHS:
        print("GETTING MODEL" + model_str)
        model = get_model(model_str)
        for test_path in sorted(listdir(TestPaths.PATH)):
            print("RUNNING TEST" + test_path)
            test_path = join(TestPaths.PATH, test_path)
            test = get_test(test_path)
            sequences = sequence_from_data(test)
            reconstructed_sequences = model.predict(sequences, batch_size = 4)
            sequences_reconstruction_cost = np.array(
                [np.linalg.norm(
                    np.subtract(sequences[i],reconstructed_sequences[i]))
                    for i in range(0,test.shape[0] - 10)])
            result = testResult.from_inference_cost(model_str, test_path, sequences_reconstruction_cost)
            result.save_to_file(results_folder)

def evaulate_from_results():
    with open(Config.RESULTS_FILE, 'r') as f:
        results = json.load(f)
    for result in results:
        path = result['Path']


if __name__ == "__main__":
    evaluate()
