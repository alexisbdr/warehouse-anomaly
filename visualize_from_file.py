import sys
from os import listdir
from os.path import join, isdir
import cv2
import json
import matplotlib.pyplot as plt

from result_dataclass import testResult

results_folder = "Results/DSCWillmer"


def run_tests():

    for file in sorted(listdir(results_folder), reverse=False):
        with open(join(results_folder,file), 'r') as f:
            json_dict = json.load(f)
            for model in json_dict:
                result = testResult.from_dict(json_dict[model], model)
                path = result.test_path
                frames = result.test_frames
                ax = result.setup_plot()

                def drawing_loop():
                    """
                    Uses cv2 to show the image from the current idx
                    """
                    skip = False
                    idx = 0; c = 0
                    w = 288; h = 162
                    imgdata = cv2.imread(join(path,frames[idx]))
                    imgdata = cv2.resize(imgdata, (w * 3, h * 3), cv2.INTER_AREA)
                    #cv2.imshow(f"{frames[idx]}", imgdata)
                    points = []
                    points.extend(ax.plot(idx, result.sequence_score[idx], '.b-'))
                    try:
                        while idx < len(result.sequence_score) + 10:
                            print(idx)
                            cv2.imshow(f'{frames[idx]}', imgdata)
                            cv2.moveWindow(f'{frames[idx]}', 40, 30)
                            c = cv2.waitKey()
                            if idx >= 10: [p.remove() for p in points]

                            #Set boundary
                            if idx < 0: idx = 0

                            new_idx = idx
                            if c == 13 or c == 100:
                                new_idx += 1
                            elif c == 8 or c == 97:
                                new_idx -= 1
                            elif c == ord('q'):
                                skip = True
                            else:
                                cv2.putText(imgdata,
                                    "Press Enter to go forward and Backspace to go back",
                                    (40, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    .8,
                                    (40, 40, 255),
                                    2)

                            if idx != new_idx:
                                #Index change - update the frame and set the index

                                idx = new_idx
                                imgdata = cv2.imread(join(path,frames[idx]))

                                imgdata = cv2.resize(imgdata, (w * 3, h * 3), cv2.INTER_AREA)

                            if idx >= 10:
                                points = []
                                for i in range(idx - 10):
                                    frame_score = result.sequence_score[i]
                                    points.extend(ax.plot(i, frame_score, '.b-'))
                                plt.pause(0.01)

                            if skip:
                                cv2.destroyAllWindows()
                                plt.close()
                                return

                            cv2.destroyAllWindows()

                    except KeyboardInterrupt:
                        sys.exit(0)

                    plt.close()
                    cv2.destroyAllWindows()
                    return

                drawing_loop()


run_tests()
