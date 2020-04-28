from os import listdir
from os.path import join, isdir
import cv2
import json
import matplotlib.pyplot as plt

from result_dataclass import testResult

results_folder = "Results/DSCWillmer"

def run_tests():

    for file in sorted(listdir(results_folder), reverse=True):
        with open(join(results_folder,file), 'r') as f:
            json_dict = json.load(f)
            for model in json_dict:
                result = testResult.from_dict(json_dict[model], model)
                path = result.test_path
                frames = result.test_frames
                ax = result.setup_plot()

                for idx in range(len(result.sequence_score)):
                    imgdata = cv2.imread(join(path,frames[idx]))
                    imgdata = cv2.resize(imgdata, (512, 512), cv2.INTER_AREA)
                    cv2.imshow('frame', imgdata)
                    frame_score = result.sequence_score[idx]
                    ax.plot(idx, frame_score, ".b-")
                    c = cv2.waitKeyEx(0)
                    print(c)
                    plt.pause(.01)
                plt.show()

run_tests()
