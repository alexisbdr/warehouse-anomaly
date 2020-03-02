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
    UCSD1 =  [
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test027",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test019",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test014",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test030"]
    UCSD2 = [
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test002",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test004",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test007",
        "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test011"]
    Avenue = [
        "data/Avenue_Dataset/testing_videos/Test010",
        "data/Avenue_Dataset/testing_videos/Test005",
        "data/Avenue_Dataset/testing_videos/Test006",
        "data/Avenue_Dataset/testing_videos/Test009"]
    ALL = [UCSD1, UCSD2, Avenue]
    
        
class Config:
    RESULTS_FILE = "test_results_all.csv"
    BASE_MODEL_PATH = "model_lstm.hdf5"
    UCSD_MODEL_PATH = "UCSD_multi_model_lstm.hdf5"
    MULTI_MODEL_PATH = "UCSD+Avenue_model_lstm.hdf5"
    MODEL_PATHS = [
        BASE_MODEL_PATH,
        UCSD_MODEL_PATH]
        #MULTI_MODEL_PATH]

def get_model(model):
    tf.keras.backend.set_floatx('float32')
    return load_model(
        model, 
        custom_objects = {'LayerNormalization': LayerNormalization})

def get_test(test_path: str):
    sz = len([name for name in listdir(test_path) if name.endswith('tif') or name.endswith('png')])
    test = np.zeros(shape=(sz, 256, 256, 1))
    cnt = 0
    for f in sorted(listdir(test_path)):
        if str(join(test_path, f))[-3:] == "tif":
            #img = cv2.imread(join(Config.TEST_PATHS[0], f))
            #img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
            img = Image.open(join(test_path, f)).resize((256, 256))
        elif  str(join(test_path, f))[-3:] == "png":
            img = Image.open(join(test_path, f)).resize((256, 256)).convert('L')
        else: continue
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
    """ FOR HOLDING THE WHOLE CSV IN ONE PLACE
    fieldnames = ["Model Path", "Test Path", "Total Frames", "Sequence Cost", "Score"]
    with open(Config.RESULTS_FILE, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    """
    for model_str in Config.MODEL_PATHS:
        print("GETTING MODEL" + model_str)
        model = get_model(model_str)
        for test_list in TestPaths.ALL:
            for test_path in test_list:
                print("RUNNING TEST" + test_path)
                if test_path != "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test027":
                    continue
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
            
                if test_path in TestPaths.UCSD1: file_id = "UCSD1"
                elif test_path in TestPaths.UCSD2: file_id = "UCSD2"
                elif test_path in TestPaths.Avenue: file_id = "Avenue"

                results_file = "Results/" + file_id + "_" + test_path.split("/")[-1] + ".json"
                
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
    for file in sorted(listdir("Results")):
        with open(file, 'r') as f:
            results = json.load(f)
            model = result['model']
            path = result['Path']
            frames = [f for f in listdir(path) if f.endswith('.tif') or f.endswith('png')]
            
            plt.axis([0,len(result['Score']),min(result['Score']) - .1,1.0])
            fg, (ax1, ax2, ax3) = plt.subplot(1,3, sharey=True)
            
            for idx, frame_score in enumerate(result["Score"]):
                imgdata = cv2.imread(join(path,frames[idx]))
                imgdata = cv2.resize(imgdata, (512, 512), cv2.INTER_AREA)
                cv2.imshow('frame', imgdata)
                plt.plot(idx, frame_score, ".b-")
                plt.pause(.02)
            plt.show()
        
evaluate()


