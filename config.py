import os

# Base paths for the Project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =========================================================================
# User-Defined Data Paths
# =========================================================================
# Change this path to where your RAW THINGS-MEG data is stored!
MEG_RAW_DIR = "/mnt/d/THINGSdata/THINGS-MEG/"
# Change this path to where you want your PREPROCESSED THINGS-MEG data to be stored!
MEG_PREPROCESSED_DIR = "/mnt/d/THINGSdata/THINGS-MEG/derivatives/preprocessed/"

# Path mappings for intermediate/processed data
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SENSOR_POS_DIR = os.path.join(DATA_DIR, "sensor_positions")
EXTRACTED_FEATURES_DIR = os.path.join(DATA_DIR, "extracted_features")
