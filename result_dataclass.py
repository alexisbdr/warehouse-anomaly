from os import listdir
import matplotlib.pyplot as plt

from dataclasses import dataclass, field

annotation_mapping = {
    'Test Path': 'test_path',
    'Total Frames': 'size',
    'Sequence Cost': 'sequence_cost',
    'Score': 'sequence_score'
}

@dataclass
class testResult:

    model: str = ""
    test_path: str = ""
    size: int = 0

    sequence_cost: list = field(default_factory=list)
    sequence_score: list = field(default_factory=list)

    @property
    def test_frames(self):
        frames = []
        for frame in listdir(self.test_path):
            if frame.endswith(('.tif','.jpeg','.png')):
                frames.append(frame)
        return frames

    def setup_plot(self):
        """
        returns a matplotlib subplot object that you can call plot() on
        """
        plt.axis([0,len(self.sequence_score),min(self.sequence_score) - .1,1.0])
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.plot([], ".b-", label=self.model.strip('.hdf5'))
        ax.legend(loc='center left', bbox_to_anchor=(.8, 0.9))

        return ax

    def save_to_json(folder: str):

        if not self.test_path:
            raise Exception("Cannot save file - path to test folder has not been set")

        test_name = self.test_path.split("/")[-1]
        results_file = f"{folder}_{test_name}.json"
        out = self.asdict()

        if exists(results_file):
            print("APPENDING TO FILE: " + results_file)
            with open(results_file, mode="r+") as json_file:
                data = json.load(json_file)
                data[self.model] = out
                json_file.seek(0)
                json.dump(data, json_file)
        else:
            with open(results_file, mode='w') as json_file:
                print("CREATED NEW FILE: " + results_file)
                out_data = {}
                out_data[self.model] = out
                json.dump(out_data, json_file)

    @classmethod
    def from_dict(cls, dict_in, model):
        """
        Expects a python dict object and the key which is the model name
        Old files were saved in a different json format - we will have to account for that here
        """
        mod = dict_in.copy()
        use_dict = {}
        use_dict['model'] = model
        for entry in mod:
            if entry in cls.__annotations__:
                use_dict[entry] = mod[entry]
            elif entry in annotation_mapping.keys():
                use_dict[annotation_mapping[entry]] = mod[entry]
        return cls(**use_dict)

    @classmethod
    def from_inference_cost(cls, model, path, reconstruction_cost):

        sa = (reconstruction_cost - np.min(reconstruction_cost)) / np.max(reconstruction_cost)
        sr = 1.0 - sa
        num_frames = sr.shape

        result = cls(
            model_name = model,
            test_path = path,
            size = num_frames,
            sequence_cost = reconstruction_cost,
            sequence_score = sr
        )
        return result


