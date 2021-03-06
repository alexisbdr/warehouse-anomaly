# Warehouse Anomaly Detection

## Problem Statement
### Warehouses
Warehouses can become very messy and unorganized, objects lying around can put workers at risk or some of the merchandise can get damaged. workers are often too busy to spot these anomalies
I want to find a way to automatically flag these abnormalities and report them to the warehouse staff. Such as: 
- propane tanks lying around
- damaged boxes
- debris
- phone usage
- liquid spills
Our aim is to detect these using a single approach that can flag when there is an "anomaly"

### Anomaly Detection in video data
Anomaly detection

### Input & Output
Input: sequence of images from video
Output: a regularity score at each frame based off of the euclidean distance between the reconstruction and the actual frame

## Network architecture
The approach is to teach an autoencoder regularity from a sequence of images. We use Convolutional LSTM to extract spatial features using the convolutional layers and temporal dependencies between frames using the LSTM layers.

You can read these papers for more information: 
* Yong Shean Chong, Abnormal Event Detection in Videos using Spatiotemporal Autoencoder (2017), https://arxiv.org/abs/1701.01546
* Mahmudul Hasan, Jonghyun Choi, Jan Neumann, Amit K. Roy-Chowdhury, Learning Temporal Regularity in Video Sequences (2016), https://arxiv.org/abs/1604.04574


## Run Anomaly Detection
### Setting up the environment
#### OS/Hardware Requirements
* [Ubuntu 16+ (Tested on Ubuntu 18.04)]
* [Nvidia GPU w/ 4Gb+ Memory]
* [~50GB of SSD storage]
* [Tensorflow-gpu 2.0+]
* [OpenCV3]

Follow these steps:

#### [SUGGESTED] EC2 Instance & Deep Learning AMI
* [Deep Learning Base AMI](https://aws.amazon.com/marketplace/pp/B07Y3VDBNS)
* Instance Type:
    * testing: g4dn.xlarge
    * training: g4dn.4xlarge
* Storage: 
    * 50Gb of ssd

### OPTION 1 - [DOCKER](https://hub.docker.com/repository/docker/abaudron0215/warehouse-anomaly)
```
docker pull abaudron0215/warehouse-anomaly
docker run --gpus all -it abaudron0215/warehouse-anomaly bash
```

### OPTION 2 - MANUAL INSTALL WITH ANACONDA
#### 1. Install Anaconda for Ubuntu
```
## You can visit (https://www.anaconda.com/distribution/) to install a different version of Anaconda
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

## Check the sum 
sha256sum Anaconda3-2020.02-Linux-x86_64.sh

## Run the script and answer 'yes' to everything
bash Anaconda3-2020.02-Linux-x86_64.sh
```

#### 2. Create and activate the Tensorflow environment
```
source ~/.bashrc
conda create --name tf-gpu tensorflow-gpu
conda activate tf-gpu
conda install pillow matplotlib
```
#### 3. Clone the repo
```
git clone https://github.com/alexisbdr/warehouse-anomaly
```
#### 4. Download data and models - see Available Resources section below 

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

## Available Resources (data and models)

You do not need to do the following if you insalled via Docker
To download large files from Google Drive:
* Download Gdown
    * Anaconda
    ```
    conda install -c conda-forge gdown
    ```
    * Pip
    ```
    pip install gdown
    ```

## 1. Data

### Pepared Datasets

```
#Navigate to the repo
cd warehouse-anomaly
source download_data.sh
```

You will find two folders in the downloaded "data/" folder:
* UCSD Pedestrian: Video sequences of pedestrians on a walkway
* CUHK Avenue: Vdieo sequence of people at a subway entrance

### Your Own Data

If you want to train/test the model with your own data you change the paths in config.py to your own folders. The data should be split into a sequence of images using ffmpeg
```
#Install ffmpeg
sudo apt-get install ffmpeg
ffmpeg -i path_to_vid.mp4 -frames 400 -r 10 Test/img%04.tif
```
You should play with the frames and r (framerate) parameters to get the most information out of the video sequenc

## 2. Models

### Downloading Pre-Trained Models

Model Name  | Trained on | Drive Link | Gdown link
------------- | ------------- | -------------------------------- | --------------------------------
model_lstm.hdf5  | UCSD Ped 1 | [link](https://drive.google.com/open?id=1eIKttsHFUlZdWI4k7rIMDDsOAgURkhTp) | [link](https://drive.google.com/uc?id=1eIKttsHFUlZdWI4k7rIMDDsOAgURkhTp)
UCSD_multi_model_lstm.hdf5  | UCSD Ped 1 & 2 | [link](https://drive.google.com/open?id=19L5mcQk3CllZRfv3iIErdoVhNcX9s50m) | [link](https://drive.google.com/uc?id=19L5mcQk3CllZRfv3iIErdoVhNcX9s50m)
UCSD+Avenue_model_lstm.hdf5 | UCSD Ped 1 & 2 / CUHK Avenue | [link](https://drive.google.com/open?id=1BCTHVZc4FnjveEcxEVU9DIMgfwJdsjx9) | [link](https://drive.google.com/uc?id=1BCTHVZc4FnjveEcxEVU9DIMgfwJdsjx9)

#### Command line instructions

```
cd ~/warehouse-anomaly
#e.g downloading the first model
source download_models.sh
```

### Browser instructions
Visit a one of the "Drive link" in the table above and download file to warehosue-anomaly repo
