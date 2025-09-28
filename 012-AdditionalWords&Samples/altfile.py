import pyedflib
import mne
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import shutil
import joblib

# PLEASE ENSURE THAT FILES WILL GO TO THE FF> DIRECTORIES. OTHERWISE, IT WILL NOT READ
reference_dir = r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\012-AdditionalWords&Samples\Recordings\REFERENCES"
samples_dir = r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\012-AdditionalWords&Samples\Recordings\SAMPLES"
model_path = os.path.join(reference_dir, 'edf_classifier_model.pkl')

#CONSTANTS
MAX_FEATURE_LENGTH = 200  # Maximum length, adjust if necessary; Required by MNE
DEFAULT_LABELS = ['yes', 'no', 'hello', 'please', 'sorry', 'thanks']    #Do note that these could still have more labels added via 'new' function

#Load the list of "interpretations" based on existing directory folders from 'REFERENCES'
LABELS = [folder.lower() for folder in os.listdir(reference_dir) if os.path.isdir(os.path.join(reference_dir, folder))]

#THIS IS FOR FEATURE EXTRACTION USING THE MNE LIBRARY; MOSTLY AUTOMATED
def load_edf_features(file_path):
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        print(f"File not found: {absolute_path}")
        return None
    
    raw = mne.io.read_raw_edf(absolute_path, preload=True, verbose=False)
    data, _ = raw[:]
    
    features = []
    try:
        for channel in data:
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.median(channel))
            features.append(np.max(channel))
            features.append(np.min(channel))
            features.append(np.sum(np.square(channel)) / len(channel))  # Signal Power
    except Exception as e:
        print(f"Failed to process channel in {file_path}: {e}")
        return None
    
    features = np.array(features)
    if len(features) < MAX_FEATURE_LENGTH:
        features = np.pad(features, (0, MAX_FEATURE_LENGTH - len(features)), mode='constant')
    else:
        features = features[:MAX_FEATURE_LENGTH]
    
    return features

#FUNCTION FOR 'new' MENU OPTION; ADDS NEW WORD/FOLDER ON 'REFERENCES' DIRECTORY
def add_new_label():
    global LABELS
    new_label = input("What is the new word?: ").strip().lower()
    new_label_dir = os.path.join(reference_dir, new_label)
    os.makedirs(new_label_dir, exist_ok=True)
    sample_file = input("Enter the name of the .edf file to add: ")
    sample_path = os.path.join(samples_dir, sample_file)
    shutil.copy(sample_path, new_label_dir)
    LABELS.append(new_label)
    print(f"File added to {new_label} folder. Training the model...")
    train_and_save_model()

#FUNCTION FOR 'add' MENU OPTION; ADD MORE .edf FILES TO A REF FOLDER
def add_to_existing_label():
    global LABELS
    print(f"Available labels: {', '.join(LABELS)}")
    selected_label = input("Select a label: ").strip().lower()
    if selected_label in LABELS:
        sample_file = input("Enter the name of the .edf file to add: ")
        sample_path = os.path.join(samples_dir, sample_file)
        shutil.copy(sample_path, os.path.join(reference_dir, selected_label))
        print(f"File added to {selected_label} folder. Training the model...")
        train_and_save_model()

#FUNCTION FOR 'reset' MENU OPTION; RESETS THE ML MODEL
def reset_model():
    """Reset the current model and dataset."""
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Model reset successfully.")
    else:
        print("No model found to reset.")

#FUNCTION FOR 'remove' MENU OPTION; REMOVES FOLDER/INTERPRETATION 
def remove_label():
    """Remove a label, its contents, and update references without deleting sample files."""
def remove_label():
    global LABELS
    print(f"Available labels: {', '.join(LABELS)}")
    selected_label = input("Enter the label to remove: ").strip().lower()
    selected_label_dir = os.path.join(reference_dir, selected_label)
    if os.path.exists(selected_label_dir):
        shutil.rmtree(selected_label_dir)
        LABELS.remove(selected_label)
        print(f"Label '{selected_label}' removed. Retraining the model...")
        train_and_save_model()
    else:
        print(f"Label '{selected_label}' does not exist.")

#THIS IS FOR LOCATING DATASE AND USING THE 'REFERENCES' DIRECTORY
def load_training_data():
    #LOAD TRAINING DATA FROM REF EDF FILES; APPEND SAVED DATASET
    X, y = [], []
    for label in LABELS:
        folder_path = os.path.join(reference_dir, label)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.edf'):
                file_path = os.path.join(folder_path, file_name)
                features = load_edf_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        if 'X' in model_data and 'y' in model_data:
            X.extend(model_data['X'])
            y.extend(model_data['y'])
    
    if len(X) == 0:
        raise ValueError("No valid EDF feature data found. Cannot train the model.")
    
    return np.array(X), np.array(y)

#MNE SYNTAXES FOR APPLYING MACHINE LEARNING IN 'edf_classifier_model.pkl'
def train_and_save_model():
    X, y = load_training_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump({'model': model, 'X': X, 'y': y}, model_path)
    print("Model trained and saved successfully.")


#CLASSIFICATION LOGIC FOR ML MODEL
def classify_edf(new_file):
    """Classify a new EDF file."""
    new_features = load_edf_features(new_file)
    if new_features is None:
        return "File not found or invalid features"
    
    if not os.path.exists(model_path):
        print("Model does not exist. Training model now...")
        train_and_save_model()
    
    model_data = joblib.load(model_path)
    
    #Check if model_data is a dictionary or directly a model
    if isinstance(model_data, dict):
        model = model_data['model']
    else:
        model = model_data  #Assume it was saved directly as a model
    
    prediction = model.predict([new_features])
    return prediction[0].lower()

#THIS IS FOR RETRAINING THE MODEL USING "SAMPLE" FILES (i.e. using terminal)
def retrain_model_with_new_data(new_features, correct_label):
    """Retrain the model with additional labeled data, keeping the data consistent."""
    X, y = load_training_data()
    
    # Ensure that only the relevant label's data is retrained
    X = np.vstack([X, new_features])
    y = np.append(y, correct_label.lower())
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump({'model': model, 'X': X, 'y': y}, model_path)
    print("Model retrained and saved successfully with the new data.")

#FUNCTION FOR 'all' MENU OPTION; CHECKS INTERPRETATION FOR ALL FILES IN 'SAMPLES' DIRECTORY
def process_all_files():
    """Process all files in the SAMPLES directory."""
    for file_name in os.listdir(samples_dir):
        if file_name.endswith('.edf'):
            file_path = os.path.join(samples_dir, file_name)
            interpretation = classify_edf(file_path)
            print(f"{file_name} === {interpretation}")

#MAIN INTERFACE/INITIAL TERMINAL DISPLAY
def classify_and_train():
    while True:
        print("=================================================================")
        new_file_name = input("Enter the name of the new EDF file ('stop', 'all', 'new', 'add', 'reset', 'remove'): ")
        if new_file_name.lower() == 'stop':
            break
        if new_file_name.lower() == 'all':
            process_all_files()
            continue
        if new_file_name.lower() == 'new':
            add_new_label()
            continue
        if new_file_name.lower() == 'add':
            add_to_existing_label()
            continue
        if new_file_name.lower() == 'reset':
            reset_model()
            continue
        if new_file_name.lower() == 'remove':
            remove_label()
            continue
        new_file = os.path.join(samples_dir, new_file_name)
        response = classify_edf(new_file)
        if response == "File not found or invalid features":
            continue
        print(f"The classification is: {response}")
        
        while (correct := input("Is the interpretation correct? (yes/no): ").lower()) not in ['yes', 'no', '1', '2']:
            print("Invalid input. Please type 'yes' or 'no'.")
        if correct in ['no', '2']:
            print(f"Available labels: {', '.join(LABELS)}")
            correct_label = input("Enter the correct label: ").strip().lower()
            retrain_model_with_new_data(load_edf_features(new_file), correct_label)
        else:
            retrain_model_with_new_data(load_edf_features(new_file), response)

if __name__ == "__main__":
    classify_and_train()
