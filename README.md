# Warehouse Anomaly Detection

## Problem Statement
### Warehouses
Warehouses can become very messy and unorganized, objects lying around can put workers at risk or some of the merchandise can get damaged. workers are often too busy to spot these anomalies
I want to find a way to automatically flag these abnormalities and report them to the warehouse staff. Such as: 
-propane tanks lying around
-damaged boxes
-debris
-phone usage
-liquid spills
Our aim is to detect these using a single approach that can flag when there is an "anomaly"

### Anomaly Detection in video data
Anomaly detection


### Input & Output
We feed a sequence of images to the model and it outputs a regularity score for each frame. This score is based of it's learned reconstruction


## Run Anomaly Detection
### Setting up the environment
#### Requirements
[Ubuntu 16+ (Tested on Ubuntu 18.04)]
[Nvidia GPU w/ 4Gb+ Memory]
[Tensorflow-gpu 2.0]
[OpenCV3]

Follow these steps:

#### 1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/)
#### 2. Create and activate the Tensorflow environment
```
conda create --name tf-gpu tensorflow-gpu
conda activate tf-gpu
```
#### 3. Clone the repo
```
git clone https://github.com/alexisbdr/warehouse-anomaly
```
#### 4. Download the data - see data section below

### Testing
You can run a pretrained model on a single test path by doing the following:
```
python ucsd_det_inline.py -test
```
To change the test path you can change the path of the following variable in Config.py
```
Config.SINGLE_TEST_PATH
```

You can run the comparative benchmark that runs the pretrained models on data from 3 different datasets
```
python video_test.py
```

### Training
To train the model from scratch:
```
python ucsd_det_inline.py -train
```
By default the Config file is set to train the model using the UCSD1 Dataset path, follow the instructions in the Data section to see how to create data that can be used to train the model. 
Change the training path in Config.py

## Data

### Prepared Datasets for testing and training

https://drive.google.com/drive/folders/1W1NrZyu-Q6R481zwLT7m8bfaGGAwAPYd?usp=sharing

### Your Own Data
...


## Pre-Trained Models
<!-->
As part of the demo I trained 3 models from scratch using video sequences from 3 different datasets:
-[UCSD1](https://drive.google.com/file/d/1eIKttsHFUlZdWI4k7rIMDDsOAgURkhTp/view?usp=sharing) - trained on the UCSD Pedestrian 1 dataset
-[UCSD1 & UCSD2](https://drive.google.com/file/d/19L5mcQk3CllZRfv3iIErdoVhNcX9s50m/view?usp=sharing) - trained on both UCSD Pedestrian 1 & 2 
-[UCSD1 & UCSD2 & CUHK](https://drive.google.com/file/d/1BCTHVZc4FnjveEcxEVU9DIMgfwJdsjx9/view?usp=sharing) - trained on UCSD1, UCSD2 and the CUHK Avenue dataset
-->