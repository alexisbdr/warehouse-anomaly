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

                def drawing_loop(idx):
                    """
                    Uses cv2 to show the image from the current idx
                    """
                    if idx < 0:
                        return

                    w = 288
                    h = 162
                    imgdata = cv2.imread(join(path,frames[idx]))
                    imgdata = cv2.resize(imgdata, (w * 3, h * 3), cv2.INTER_AREA)

                    cv2.imshow('frame', imgdata)
                    while True:
                        points = []
                        for i in range(idx):
                            frame_score = result.sequence_score[i]
                            points.extend(ax.plot(i, frame_score, '.b-'))
                        plt.pause(0.01)

                        c = cv2.waitKeyEx()
                        if c == 13 or c == 100:
                            [p.remove() for p in points]
                            drawing_loop(idx+1)
                        elif c == 8 or c == 97:
                            [p.remove() for p in points]
                            drawing_loop(idx - 1)
                        else:
                            cv2.putText(imgdata,
                                        "Press Enter to go forward and Backspace to go back",
                                        (40, 250),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        .8,
                                        (40, 40, 255),
                                        2)
                            cv2.imshow('frame', imgdata)

                drawing_loop(0)

run_tests()
