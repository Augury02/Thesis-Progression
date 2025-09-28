import pyedflib
import mne
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

# PLEASE ENSURE THAT FILES WILL GO TO THE FF> DIRECTORIES. OTHERWISE, IT WILL NOT READ
reference_dir = r"C:\Users\jayve\Downloads\Kent\Coding FIles\006.1_DryRun\Recordings\REFERENCES"
samples_dir = r"C:\Users\jayve\Downloads\Kent\Coding FIles\006.1_DryRun\Recordings\SAMPLES"
model_path = os.path.join(reference_dir, 'edf_classifier_model.pkl')

# --- Constants ---
MAX_FEATURE_LENGTH = 200  # Arbitrary maximum length, adjust if necessary

def load_edf_features(file_path):
    """Extract features from an EDF file."""
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"File not found: {absolute_path}")
    
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
    
    # Ensure consistent feature length
    features = np.array(features)
    if len(features) < MAX_FEATURE_LENGTH:
        # Pad with zeros
        features = np.pad(features, (0, MAX_FEATURE_LENGTH - len(features)), mode='constant')
    else:
        # Truncate if too long
        features = features[:MAX_FEATURE_LENGTH]
    
    return features

def load_training_data():
    """Load training data from reference EDF files."""
    X, y = [], []
    labels = ['yes', 'no', 'hello', 'please', 'sorry', 'thanks']
    
    for label in labels:
        folder_path = os.path.join(reference_dir, label.capitalize())
        if not os.path.exists(folder_path):
            print(f"Warning: Folder for label '{label}' not found at {folder_path}")
            continue
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.edf'):
                file_path = os.path.join(folder_path, file_name)
                try:
                    features = load_edf_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(label)
                except FileNotFoundError as e:
                    print(e)
                except Exception as e:
                    print(f"Failed to load features from {file_path}: {e}")
    
    if len(X) == 0:
        raise ValueError("No valid EDF feature data found. Cannot train the model.")
    
    return np.array(X), np.array(y)

def train_and_save_model():
    """Train and save the model."""
    X, y = load_training_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_path)
    print("Model trained and saved successfully.")

def classify_edf(new_file):
    """Classify a new EDF file."""
    try:
        new_features = load_edf_features(new_file)
        if new_features is None:
            return "Invalid EDF file or feature extraction failed."
    except FileNotFoundError as e:
        print(e)
        return "File not found"
    
    if not os.path.exists(model_path):
        print("Model does not exist. Training model now...")
        train_and_save_model()
    
    model = joblib.load(model_path)
    prediction = model.predict([new_features])
    return prediction[0]

def retrain_model_with_new_data(new_features, correct_label):
    """Retrain the model with additional labeled data."""
    try:
        X, y = load_training_data()
        X = np.vstack([X, new_features])
        y = np.append(y, correct_label)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        print("Model retrained and saved successfully with the new data.")
    except Exception as e:
        print(f"Failed to retrain the model: {e}")

def classify_and_train():
    while True:
        # Ask for new file to classify
        new_file_name = input("Enter the name of the new EDF file from SAMPLES (or type 'stop' to exit): ")
        
        if new_file_name.lower() == 'stop':
            print("Exiting the program.")
            break
        
        new_file = os.path.join(samples_dir, new_file_name)
        print("Classifying the new EDF file...")

        response = classify_edf(new_file)
        print(f"The initial classification of the file is: {response}")

        # Ask if classification was correct
        correct = input(f"Is the interpretation of '{new_file_name}' correct? (yes/no): ").lower()
        
        if correct == 'yes':
            retrain = input("Would you like to use this data to retrain the model? (yes/no): ").lower()
            if retrain == 'yes':
                print("Retraining the model with the current data...")
                new_features = load_edf_features(new_file)
                retrain_model_with_new_data(new_features, response)
            else:
                print("Model not retrained.")
        else:
            correct_label = input("What is the correct label for this file? (yes/no/hello/please/sorry/thanks): ").lower()
            if correct_label not in ['yes', 'no', 'hello', 'please', 'sorry', 'thanks']:
                print("Invalid label. Skipping retraining.")
                continue
            print("Retraining the model with the corrected data...")
            new_features = load_edf_features(new_file)
            retrain_model_with_new_data(new_features, correct_label)

if __name__ == "__main__":
    classify_and_train()
