from pathlib import Path

DATASET = "cosmos"
LABELS_FILE = "labels.csv"
SAMPLE_RATE = 44100
CHUNK_LENGTH = 5 #seconds
FFT_SIZE = 1024
NUM_MEL_FILTERBANKS = 224

DATASET_DIR = Path.cwd().parent/"datasets"/DATASET
ORIGINAL_DATA_DIR = DATASET_DIR/(DATASET+"_original")
CHUNKED_DATA_DIR = DATASET_DIR/(DATASET+"_chunked")
TRAINING_DATA_DIR = CHUNKED_DATA_DIR


LABEL_MAP = {
             'No class of interest':0, 
             'bird':1, 
             'Bird':1, 
             'Celeus grammicus':2, 
             'Trogon viridis':3, 
             'Thraupis episcopus':4, 
             'Xiphorhynchus guttatus':5, 
             'Lipaugus vociferans':6, 
             'Turdus leucomelas':7, 
             'Zonotrichia capensis':8, 
             'Myioborus miniatus':9, 
             'Microcerculus marginatus':10, 
             'Ramphastos tucanus':11, 
             'Tolmomyias sulphurescens':12}
