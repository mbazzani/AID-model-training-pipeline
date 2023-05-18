from pathlib import Path

DATASET = "cosmos"

# Path conventions, probably don't change unless we are completely
# changing conventions
LABELS_FILE = "labels.csv"
DATASET_DIR = Path.cwd().parent/"datasets"/DATASET
ORIGINAL_DATA_DIR = DATASET_DIR/(DATASET+"_original")
CHUNKED_DATA_DIR = DATASET_DIR/(DATASET+"_chunked")


TRAINING_DATA_DIR = CHUNKED_DATA_DIR
SAMPLE_RATE = 44100

# If this is changed, the existing CHUNKED_DATA_DIR must be deleted
CHUNK_LENGTH = 5 #seconds


FFT_SIZE = 1024
NUM_MEL_FILTERBANKS = 224


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
