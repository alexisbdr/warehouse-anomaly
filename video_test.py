import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LayerNormalization

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
        img = np.array(img, dtype=np.float32) / float(imsize)
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
            sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
            sr = 1.0 - sa
            out = {}
            out["Test Path"] = test_path
            out["Total Frames"] = sr.shape
            out["Sequence Cost"] = sequences_reconstruction_cost.tolist()
            out["Score"] = sr.tolist()
        
            results_file = "Results/DSCWillmer" + "_" + test_path.split("/")[-1] + ".json"
            
            if exists(results_file):
                print("APPENDING TO FILE: " + results_file)
                with open(results_file, mode="r+") as json_file:
                    data = json.load(json_file)
                    data[model_str] = out
                    print(data)
                    json_file.seek(0)
                    json.dump(data, json_file)
            else:
                with open(results_file, mode='w') as json_file:
                    print("CREATED NEW FILE: " + results_file)
                    out_data = {}
                    out_data[model_str] = out
                    json.dump(out_data, json_file)
    
def evaulate_from_results():
    with open(Config.RESULTS_FILE, 'r') as f:
        results = json.load(f)
    for result in results:
        path = result['Path']


def run_tests():
    colors = [".b-", "xr-", ".g-"]
    for file in sorted(listdir("Results"), reverse=True):
        with open(join("Results",file), 'r') as f:
            results = json.load(f)
            models = list(results.keys())
            print(models)
            path = results[models[0]]['Test Path']
            print(path)
            frames = [f for f in listdir(path) if f.endswith('.tif') or f.endswith('png')]
            plt.axis([0,len(results[models[0]]['Score']),min(results[models[0]]['Score']) - .1,1.0])
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.plot([], colors[0], label=models[0].strip('.hdf5'))
            ax.plot([], colors[1], label=models[1].strip('.hdf5'))
            ax.plot([], colors[2], label=models[2].strip('.hdf5'))
            ax.legend(loc='center left', bbox_to_anchor=(.8, 0.9))
            for idx in range(len(results[models[0]]["Score"])):
                imgdata = cv2.imread(join(path,frames[idx]))
                imgdata = cv2.resize(imgdata, (512, 512), cv2.INTER_AREA)
                cv2.imshow('frame', imgdata)
                for ax_idx, model in enumerate(models):
                    frame_score = results[model]['Score'][idx] 
                    ax.plot(idx, frame_score, colors[ax_idx])
                plt.pause(.01)
            plt.show()
        
evaluate()
