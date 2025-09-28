import pyedflib
import mne
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

#PLEASE ENSURE THAT FILES WILL GO TO THE FF> DIRECTORIES. OTHERWISE, IT WILL NOT READ
"""NOTE: FILE DIRECTORY IS NOW ORGANIZED INSTEAD OF ONLY BEING ABLE TO SOURCE FROM THE SAME FOLDER"""
reference_dir = r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\005-InterpreterPrototype_ MachineLearning\EDF FILES\References"
unclassified_dir = r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\005-InterpreterPrototype_ MachineLearning\EDF FILES\Unclassified"
model_path = os.path.join(reference_dir, 'edf_classifier_model.pkl')

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
        features.append(np.median(channel))
        features.append(np.max(channel))
        features.append(np.min(channel))
        features.append(np.sum(np.square(channel)) / len(channel))  # Signal Power
    return np.array(features)

def load_training_data():
    X, y = [], []
    for label in labels:
        folder_path = os.path.join(reference_dir, label.capitalize())
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.edf'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    features = load_edf_features(file_path)
                    X.append(features)
                    y.append(label)
                except FileNotFoundError as e:
                    print(e)
    return np.array(X), np.array(y)

def train_and_save_model():
    X, y = load_training_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    print("Model trained and saved successfully.")

def classify_edf(new_file):
    try:
        new_features = load_edf_features(new_file)
    except FileNotFoundError as e:
        print(e)
        return "File not found"
    
    if not os.path.exists(model_path):
        train_and_save_model()
    
    model = joblib.load(model_path)
    prediction = model.predict([new_features])
    return prediction[0]

#FEATURE LOADING AND SETTING FOR THE MAIN .EDF REFERENCE
labels = ['yes', 'no', 'hello', 'please', 'sorry', 'thanks']

#CIN FOR THE NEW UNCLASSIFIED FILE
new_file_name = input("Enter the name of the new EDF file: ")
new_file = os.path.join(unclassified_dir, new_file_name)
print("Classifying the new EDF file...")

#RESPONSE PRODUCTION BASED ON INTERPRETATION
response = classify_edf(new_file)
print(f"The response for the given EDF file is: {response}")

"""
List of .EDF file names, for copy pasting purposes:
Test 1 - Yes_INSIGHT2_255587_2024.12.20T18.26.40+08.00.edf
Test 2 - No_INSIGHT2_255587_2024.12.20T18.29.02+08.00.edf
Hello_Sample1.edf
Hello_Sample2.edf
Sorry_Sample1.edf
Sorry_Sample2.edf
Thanks_Sample1.edf
Thanks_Sample2.edf
Please_Sample1.edf
Please_Sample2.edf
Yes_Sample1.edf
Yes_Sample2.edf
No_Sample1.edf
No_Sample2.edf
"""
