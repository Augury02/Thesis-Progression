import pyedflib
import mne
import numpy as np
from scipy.spatial.distance import euclidean
import os

#PLEASE ENSURE THAT FILES WILL GO TO THE FF> DIRECTORIES. OTHERWISE, IT WILL NOT READ
"""NOTE: FILE DIRECTORY IS NOW ORGANIZED INSTEAD OF ONLY BEING ABLE TO SOURCE FROM THE SAME FOLDER"""
reference_dir = r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\002-InterpreterPrototype_FixedDirectory\EDF FILES\References"
unclassified_dir = r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\002-InterpreterPrototype_FixedDirectory\EDF FILES\Unclassified"

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

def classify_edf(new_file, reference_features):

    #PRODUCE INTERPRETATION BASED ON SIMILARITIES OF EXTRACTED FEATURES
    try:
        new_features = load_edf_features(new_file)
    except FileNotFoundError as e:
        print(e)
        return "File not found"
    
    """NOTE: This system is streamlined/generalized to increase flexibility;
    this time, it can accomodate more than just yes/no possibilities in computation"""
    distances = {label: euclidean(new_features, features) for label, features in reference_features.items()}
    return min(distances, key=distances.get)

#FEATURE LOADING AND SETTING FOR THE MAIN .EDF REFERENCE
reference_features = {}
yes_file = os.path.join(reference_dir, "Yes_Test1_001.edf")
no_file = os.path.join(reference_dir, "No_Test2_001.edf")

print("Loading reference EDF files...")
try:
    reference_features["yes"] = load_edf_features(yes_file)
    reference_features["no"] = load_edf_features(no_file)
except FileNotFoundError as e:
    print(e)
    exit()

#CIN FOR THE NEW UNCLASSIFIED FILE
new_file_name = input("Enter the name of the new EDF file: ")
new_file = os.path.join(unclassified_dir, new_file_name)
print("Classifying the new EDF file...")

#RESPONSE PRODUCTION BASED ON INTERPRETATION
response = classify_edf(new_file, reference_features)
print(f"The response for the given EDF file is: {response}")



"""
List of .EDF file names, for copy pasting purposes:
Test 1 - Yes_INSIGHT2_255587_2024.12.20T18.26.40+08.00.edf
Test 2 - No_INSIGHT2_255587_2024.12.20T18.29.02+08.00.edf

"""

    

