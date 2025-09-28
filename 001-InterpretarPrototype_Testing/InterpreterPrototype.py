import pyedflib
import mne
import numpy as np
from scipy.spatial.distance import euclidean
import os

#FILE DIRECTORY
os.chdir(r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\001InterpretarPrototype_Testing")

def load_edf_features(file_path):
    
    #PRE-REQ FOR FEATURE EXTRACTION ((MNE)) & RESPONSE FOR FAILURE TO LOCATE
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"File not found: {absolute_path}")

    raw = mne.io.read_raw_edf(absolute_path, preload=True, verbose=False)
    data, _ = raw[:]

    #COMMANDS FOR FEATURE EXTRACTION
    features = []
    for channel in data:
        features.append(np.mean(channel))
        features.append(np.std(channel))
    return np.array(features)

def classify_edf(new_file, yes_features, no_features):
    
    #PRODUCE INTERPRETATION BASED ON SIMILARITIES OF EXTRACTED FEATURES
    try:
        new_features = load_edf_features(new_file)
    except FileNotFoundError as e:
        print(e)
        return "File not found"
    yes_distance = euclidean(new_features, yes_features)
    no_distance = euclidean(new_features, no_features)

    #BINARY SITUATION, EITHER YES/NO DEPENDING ON WHICH VALUES ARE CLOSER
    if yes_distance < no_distance:
        return "yes"
    else:
        return "no"

#FEATURE LOADING AND SETTING FOR THE MAIN .EDF REFERENCE
yes_file = "Yes_Test1_001.edf"
no_file = "No_Test2_001.edf"

print("Loading reference EDF files...")
try:
    yes_features = load_edf_features(yes_file)
    no_features = load_edf_features(no_file)
except FileNotFoundError as e:
    print(e)
    exit()

#CIN FOR THE NEW UNCLASSIFIED FILE
new_file = input("Enter the name of the new EDF file: ")
print("Classifying the new EDF file...")

#RESPONSE PRODUCTION BASED ON INTERPRETATION
response = classify_edf(new_file, yes_features, no_features)
print(f"The response for the given EDF file is: {response}")
