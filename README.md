# Warehouse Anomaly Detection

## Problem Statement
### Warehouses
Warehouses can become very messy and unorganized, objects lying around can put workers at risk or some of the merchandise can get damaged. workers are often too busy to spot these anomalies
I want to find a way to automatically flag these abnormalities and report them to the warehouse staff. Such as: 
-propane tanks lying around
-damaged boxes
### Anomaly Detection in video data

### Input & Output

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
'''
conda create --name tf-gpu tensorflow-gpu
conda activate tf-gpu
'''
#### 3. Clone the repo
'''
git clone https://github.com/alexisbdr/warehouse-anomaly
'''
#### 4. Download the data - see data section below

### Testing


## Data
### Your Own Data

### Available Datasets
[UCSD Pedestrian dataset]() 
[CUHK Avenue]()

## Pre-Trained Models
As part of the demo I trained 3 models from scratch using video sequences from 3 different datasets:
[UCSD1](https://drive.google.com/file/d/1eIKttsHFUlZdWI4k7rIMDDsOAgURkhTp/view?usp=sharing) - trained on the UCSD Pedestrian 1 dataset
[UCSD1 & UCSD2](https://drive.google.com/file/d/19L5mcQk3CllZRfv3iIErdoVhNcX9s50m/view?usp=sharing) - trained on both UCSD Pedestrian 1 & 2 
[UCSD1 & UCSD2 & CUHK](https://drive.google.com/file/d/1BCTHVZc4FnjveEcxEVU9DIMgfwJdsjx9/view?usp=sharing) - trained on UCSD1, UCSD2 and the CUHK Avenue dataset
