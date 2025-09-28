import pyedflib
import mne
import numpy as np
import queue
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os
import shutil
import joblib
from gtts import gTTS
import pygame
import time
import threading
# install by typing "pip install gtts pygame"

# PLEASE ENSURE THAT FILES WILL GO TO THE FOLLOWING DIRECTORIES. OTHERWISE, IT WILL NOT READ
reference_dir = r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\17 - FIX\Recordings\REFERENCES"
samples_dir = r"C:\Users\PC\Downloads\VSCODE FILES\PYTHON\TESTING\17 - FIX\Recordings\SAMPLES"
downloads_dir = r"C:\Users\PC\Downloads"
model_path = os.path.join(reference_dir, 'edf_classifier_model.pkl')

# CONSTANTS
MAX_FEATURE_LENGTH = 200  # Maximum length, adjust if necessary; Required by MNE
DEFAULT_LABELS = ['yes', 'no', 'hello', 'please', 'sorry', 'thanks', 'more', 'less', 'none', 'all']  # Do note that these could still have more labels added via 'new' function

# Load the list of "interpretations" based on existing directory folders from 'REFERENCES'
# Validate labels against default expected labels
LABELS = [folder.lower() for folder in os.listdir(reference_dir)
          if os.path.isdir(os.path.join(reference_dir, folder)) and folder.lower() in DEFAULT_LABELS]

scaler = StandardScaler()
tts_queue = queue.Queue()

def tts_worker():
    """Worker function to process the TTS queue efficiently."""
    while True:
        text = tts_queue.get()
        if text is None:
            break  # Exit when None is received
        print(f"TTS Processing: {text}")
        tts = gTTS(text, lang='en')
        tts.save("temp_audio.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("temp_audio.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.quit()
        os.remove("temp_audio.mp3")
        print("TTS Completed")
        tts_queue.task_done()


def enqueue_tts(text):
    """Enqueue text to be processed by the TTS worker."""
    tts_queue.put(text)


def start_tts_thread():
    """Start the TTS processing thread."""
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    return tts_thread


def stop_tts_thread():
    """Stop the TTS processing thread."""
    tts_queue.put(None)


def save_model():
    joblib.dump({'model': classifier, 'scaler': scaler, 'X': X, 'y': y}, model_path)


def load_model():
    global classifier, scaler, X, y
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        classifier = model_data['model']
        scaler = model_data['scaler']
        X = model_data['X']
        y = model_data['y']
    else:
        classifier = RandomForestClassifier()
        X = np.empty((0, MAX_FEATURE_LENGTH))
        y = np.array([])
        save_model()


def monitor_directory(directory, stop_event, processed_files):
    """Monitor a directory for new EDF files and queue interpretations for speech."""
    print("Automated mode active. Scanning for new .edf files... (Type 'stop' to exit, 'back' to return)")

    while not stop_event.is_set():
        time.sleep(3)  # Wait 3 seconds before each scan
        current_files = set(f for f in os.listdir(directory) if f.endswith('.edf'))
        new_files = sorted(current_files - processed_files)  # Get new files

        if new_files:
            print("Processing new EDF files...")

            for file_name in new_files:
                if stop_event.is_set():  # Check if user wants to stop
                    break

                file_path = os.path.join(directory, file_name)
                interpretation = classify_edf(file_path)

                print(f"{file_name} === {interpretation}")

                tts_queue.put(interpretation)  # Add to queue for speaking

                processed_files.add(file_name)
                time.sleep(1)  # Small delay

            print("Resuming scan for new .edf files...")
        else:
            print("No new .edf files detected, retrying...")


def read_files_automated():
    """Automatically monitor and read newly added EDF files."""
    stop_event = threading.Event()
    processed_files = set(f for f in os.listdir(downloads_dir) if f.endswith('.edf'))

    monitor_thread = threading.Thread(target=monitor_directory, args=(downloads_dir, stop_event, processed_files), daemon=True)
    monitor_thread.start()

    tts_thread = start_tts_thread()

    while True:
        command = input("(Type 'stop' to exit, 'back' to return) > ").strip().lower()
        if command == "stop":
            print("Exiting automated mode...")
            stop_event.set()
            monitor_thread.join()
            stop_tts_thread()  # Stop the TTS thread
            tts_thread.join()
            break
        if command == "back":
            print("Returning to main menu...")
            stop_event.set()
            monitor_thread.join()
            stop_tts_thread()  # Stop the TTS thread
            tts_thread.join()
            break


def read_files_manual():
    """Manually read EDF files by user input."""
    tts_thread = start_tts_thread()

    while True:
        file_names = input("Name the file(s) to be read (comma-separated) or type 'back' to return, 'stop' to exit: ").strip().lower()

        if file_names == "stop":
            stop_tts_thread()
            tts_thread.join()
            return
        if file_names == "back":
            stop_tts_thread()
            tts_thread.join()
            break

        file_list = [f.strip() for f in file_names.split(',')]

        for file_name in file_list:
            file_path = os.path.join(samples_dir, file_name)
            if os.path.exists(file_path):
                interpretation = classify_edf(file_path)
                print(f"{file_name} === {interpretation}")
                enqueue_tts(interpretation)  # Speak the result
            else:
                print(f"File not found: {file_name}")


# THIS IS FOR FEATURE EXTRACTION USING THE MNE LIBRARY; MOSTLY AUTOMATED
def load_edf_features(file_path):
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        print(f"File not found: {absolute_path}")
        return None

    raw = mne.io.read_raw_edf(absolute_path, preload=True, verbose=False)
    data, _ = raw[:]
    session_id = os.path.basename(file_path).split('_')[0]  # Extract session ID

    features = []
    try:
        for channel in data:
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.max(channel),
                np.min(channel),
                np.sum(np.square(channel)) / len(channel),  # Signal Power
                np.percentile(channel, 25),  # 25th percentile
                np.percentile(channel, 75),  # 75th percentile
                np.var(channel),  # Variance
                np.ptp(channel)  # Peak-to-peak amplitude
            ])
    except Exception as e:
        print(f"Failed to process channel in {file_path}: {e}")
        return None

    features = np.array(features)
    if len(features) < MAX_FEATURE_LENGTH:
        features = np.pad(features, (0, MAX_FEATURE_LENGTH - len(features)), mode='constant')
    else:
        features = features[:MAX_FEATURE_LENGTH]

    return np.append(features, hash(session_id) % 1000)  # Add session ID as a numeric feature


# FUNCTION FOR 'new' MENU OPTION; ADDS NEW WORD/FOLDER ON 'REFERENCES' DIRECTORY
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


# FUNCTION FOR 'add' MENU OPTION; ADD MORE .edf FILES TO A REF FOLDER
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


# FUNCTION FOR 'reset' MENU OPTION; RESETS THE ML MODEL
def reset_model():
    """Reset the current model and dataset."""
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Model reset successfully.")
    else:
        print("No model found to reset.")


# FUNCTION FOR 'remove' MENU OPTION; REMOVES FOLDER/INTERPRETATION
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


# THIS IS FOR LOCATING DATASE AND USING THE 'REFERENCES' DIRECTORY
def load_training_data():
    """Load training data from the reference directory."""
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

    if len(X) == 0:
        raise ValueError("No valid EDF feature data found. Cannot train the model.")

    X = scaler.fit_transform(X)
    return np.array(X), np.array(y)


# MNE SYNTAXES FOR APPLYING MACHINE LEARNING IN 'edf_classifier_model.pkl'
def train_and_save_model():
    """Train and save the model."""
    X, y = load_training_data()
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

    skf = StratifiedKFold(n_splits=min(5, np.min(np.bincount([LABELS.index(label) for label in y]))))
    scores = cross_val_score(model, X, y, cv=skf)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV accuracy: {np.mean(scores):.4f}")

    model.fit(X, y)
    joblib.dump({'model': model, 'scaler': scaler, 'X': X, 'y': y}, model_path)
    print("Model trained and saved successfully.")


# CLASSIFICATION LOGIC FOR ML MODEL
def classify_edf(new_file):
    """Classify an EDF file."""
    new_features = load_edf_features(new_file)
    if new_features is None:
        return "File not found or invalid features"

    if not os.path.exists(model_path):
        print("Model does not exist. Training model now...")
        train_and_save_model()

    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']

    new_features = scaler.transform([new_features])
    prediction = model.predict(new_features)
    return prediction[0].lower()


# THIS IS FOR RETRAINING THE MODEL USING "SAMPLE" FILES (i.e. using terminal)
def retrain_model_with_new_data(new_features, correct_label):
    """Retrain the model with new labeled data."""
    model_data = joblib.load(model_path)
    X, y = model_data['X'], model_data['y']

    X = np.vstack([X, scaler.transform([new_features])])
    y = np.append(y, correct_label.lower())

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X, y)

    joblib.dump({'model': model, 'scaler': scaler, 'X': X, 'y': y}, model_path)
    print("Model retrained and saved successfully.")


# FUNCTION FOR 'all' MENU OPTION; CHECKS INTERPRETATION FOR ALL FILES IN 'SAMPLES' DIRECTORY
def process_all_files():
    """Process all files in the SAMPLES directory."""
    for file_name in os.listdir(samples_dir):
        if file_name.endswith('.edf'):
            file_path = os.path.join(samples_dir, file_name)
            interpretation = classify_edf(file_path)
            print(f"{file_name} === {interpretation}")


def validate_model_predictions():
    predictions = []
    for file_name in os.listdir(samples_dir):
        if file_name.endswith('.edf'):
            file_path = os.path.join(samples_dir, file_name)
            prediction = classify_edf(file_path)
            predictions.append(prediction)

    from collections import Counter
    print("Prediction distribution:", Counter(predictions))


# MAIN INTERFACE/INITIAL TERMINAL DISPLAY
def classify_and_train():
    load_model()

    while True:
        print("Which mode would you like to boot the program?")
        print("1. Manual File Location")
        print("2. Automated File Location")
        print("3. Debugging")
        print("4. Stop")
        mode = input("Enter mode: ").strip().lower()

        if mode == "1":
            read_files_manual()
            continue
        if mode == "2":
            read_files_automated()
            continue
        if mode == "3":
            while True:
                print("=================================================================")
                new_file_name = input("Enter EDF file name ('all', 'new', 'add', 'reset', 'remove', 'back', 'stop'): ")

                if new_file_name.lower() == 'stop':
                    print("Exiting program...")
                    return
                if new_file_name.lower() == 'back':
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

                while (correct := input("Is the interpretation correct? (yes/no): ").lower()) not in ['yes', 'no']:
                    print("Invalid input. Please type 'yes' or 'no'.")

                if correct == 'no':
                    print(f"Available labels: {', '.join(LABELS)}")
                    correct_label = input("Enter the correct label: ").strip().lower()
                    retrain_model_with_new_data(load_edf_features(new_file), correct_label)
                else:
                    retrain_model_with_new_data(load_edf_features(new_file), response)
        if mode == "4" or mode == "stop":
            print("Exiting program...")
            break
        else:
            print("Invalid mode. Please enter '1', '2', '3', or '4'.")


if __name__ == "__main__":
    load_model()
    classify_and_train()