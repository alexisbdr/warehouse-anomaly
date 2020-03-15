class Config:
  
  DATASET_PATH = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
  
  UCSD_DATASET_PATHS = ["data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train", "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"]
  
  AVENUE_DATASET_PATH = "data/Avenue_Dataset/training_videos"
  
  TRAINING_PATH = [DATASET_PATH]
  
  SINGLE_TEST_PATH = "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test002"
  
  BATCH_SIZE = 4
  
  EPOCHS = 3
  
  MODEL_PATH = "model_lstm.hdf5"


