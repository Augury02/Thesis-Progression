import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import queue
import pygame
import time
from gtts import gTTS

# Import your existing script functions and variables here
from edf_classifier import (
    add_new_label,
    add_to_existing_label,
    reset_model,
    remove_label,
    classify_edf,
    load_model,
    retrain_model_with_new_data,
    load_edf_features,
    samples_dir,
    downloads_dir  # Ensure downloads_dir is imported
)

# Initialize TTS queue
tts_queue = queue.Queue()
log_queue = queue.Queue()

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
        pygame.mixer.music.set_volume(1.0)  # Assuming volume is set to maximum
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

def process_all_files_debug():
    """Process all files in the SAMPLES directory for debug mode."""
    output = []
    for file_name in os.listdir(samples_dir):
        if file_name.endswith('.edf'):
            file_path = os.path.join(samples_dir, file_name)
            interpretation = classify_edf(file_path)
            output.append(f"{file_name} === {interpretation}")
    return "\n".join(output)

#///////////////////////////////////////////////////////////////////////
class EDFAssistantApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EDF Classifier Assistant")
        self.geometry("600x400")

        self.text_size = 10
        self.tts_volume = 1.0
        self.window_size = "600x400"

        self.create_widgets()
        self.check_log_queue()

    def create_widgets(self):
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.frames = {}
        for F in (MainPage, ManualPage, AutomatedPage, DebugPage, SettingsPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def check_log_queue(self):
        while not log_queue.empty():
            log_message = log_queue.get()
            for frame in self.frames.values():
                if hasattr(frame, 'append_log'):
                    frame.append_log(log_message)
        self.after(100, self.check_log_queue)

    def update_text_size(self, size):
        self.text_size = size
        for frame in self.frames.values():
            if hasattr(frame, 'update_text_size'):
                frame.update_text_size(size)

    def update_window_size(self, size):
        self.window_size = size
        self.geometry(size)
        width, height = map(int, size.split('x'))
        scale_factor = width / 600  # Assuming 600 is the base width for scaling
        for frame in self.frames.values():
            if hasattr(frame, 'update_window_size'):
                frame.update_window_size(scale_factor)

    def update_tts_volume(self, volume):
        self.tts_volume = volume

#///////////////////////////////////////////////////////////////////////
class MainPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        self.mode_label = tk.Label(self, text="Which mode would you like to boot the program?", font=("Helvetica", self.controller.text_size))
        self.mode_label.pack(pady=10)

        self.manual_button = tk.Button(self, text="Manual File Location", command=lambda: self.controller.show_frame("ManualPage"), font=("Helvetica", self.controller.text_size))
        self.manual_button.pack(pady=5)

        self.automated_button = tk.Button(self, text="Automated File Location", command=lambda: self.controller.show_frame("AutomatedPage"), font=("Helvetica", self.controller.text_size))
        self.automated_button.pack(pady=5)

        self.debug_button = tk.Button(self, text="Debugging", command=lambda: self.controller.show_frame("DebugPage"), font=("Helvetica", self.controller.text_size))
        self.debug_button.pack(pady=5)

        self.settings_button = tk.Button(self, text="Settings", command=lambda: self.controller.show_frame("SettingsPage"), font=("Helvetica", self.controller.text_size))
        self.settings_button.pack(pady=5)

        self.stop_button = tk.Button(self, text="Stop", command=self.controller.quit, font=("Helvetica", self.controller.text_size))
        self.stop_button.pack(pady=5)

    def update_text_size(self, size):
        self.mode_label.config(font=("Helvetica", size))
        self.manual_button.config(font=("Helvetica", size))
        self.automated_button.config(font=("Helvetica", size))
        self.debug_button.config(font=("Helvetica", size))
        self.settings_button.config(font=("Helvetica", size))
        self.stop_button.config(font=("Helvetica", size))

    def update_window_size(self, scale_factor):
        self.mode_label.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.manual_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.automated_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.debug_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.settings_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.stop_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))


#///////////////////////////////////////////////////////////////////////
class ManualPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()
        self.manual_queue = queue.Queue()
        self.current_interpretation = None  # To store the current interpretation

        # Start TTS thread
        self.tts_thread = start_tts_thread()

    def create_widgets(self):
        self.manual_label = tk.Label(self, text="Manual File Location", font=("Helvetica", self.controller.text_size))
        self.manual_label.pack(pady=10)

        self.files_frame = tk.Frame(self)
        self.files_frame.pack(pady=5)

        self.files_canvas = tk.Canvas(self.files_frame, height=200)
        self.files_canvas.pack(side=tk.LEFT)

        self.scrollbar = tk.Scrollbar(self.files_frame, orient="vertical", command=self.files_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        self.scrollable_frame = tk.Frame(self.files_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.files_canvas.configure(
                scrollregion=self.files_canvas.bbox("all")
            )
        )

        self.files_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.files_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.load_files()

        self.output_text = tk.Text(self, height=10, width=70, font=("Helvetica", self.controller.text_size))
        self.output_text.pack(pady=5)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=5)

        self.view_queue_button = tk.Button(self.button_frame, text="View Queue", command=self.view_queue, font=("Helvetica", self.controller.text_size))
        self.view_queue_button.grid(row=0, column=0, padx=5)

        self.speak_queue_button = tk.Button(self.button_frame, text="Speak Queue", command=self.speak_queue, font=("Helvetica", self.controller.text_size))
        self.speak_queue_button.grid(row=0, column=1, padx=5)

        self.done_button = tk.Button(self.button_frame, text="Done", command=self.done_adding, font=("Helvetica", self.controller.text_size))
        self.done_button.grid(row=0, column=2, padx=5)

        self.back_button = tk.Button(self, text="Back", command=lambda: self.controller.show_frame("MainPage"), font=("Helvetica", self.controller.text_size))
        self.back_button.pack(pady=5)

    def load_files(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for file_name in os.listdir(samples_dir):
            if file_name.endswith('.edf'):
                file_button = tk.Button(self.scrollable_frame, text=file_name, command=lambda fn=file_name: self.process_file(fn), font=("Helvetica", self.controller.text_size))
                file_button.pack(pady=2)

    def process_file(self, file_name):
        file_path = os.path.join(samples_dir, file_name)
        if os.path.exists(file_path):
            response = classify_edf(file_path)
            self.output_text.insert(tk.END, f"File: {file_name}, Classification: {response}\n")
            self.ask_store_interpretation(response)
        else:
            self.output_text.insert(tk.END, f"File not found: {file_name}\n")

    def ask_store_interpretation(self, interpretation):
        self.current_interpretation = interpretation
        self.output_text.insert(tk.END, "Do you want to store this interpretation in the queue?\n")

        self.yes_no_frame = tk.Frame(self)
        self.yes_no_frame.pack(pady=5)

        self.yes_button = tk.Button(self.yes_no_frame, text="Yes", command=self.store_yes, font=("Helvetica", self.controller.text_size))
        self.yes_button.grid(row=0, column=0, padx=5)

        self.no_button = tk.Button(self.yes_no_frame, text="No", command=self.store_no, font=("Helvetica", self.controller.text_size))
        self.no_button.grid(row=0, column=1, padx=5)

    def store_yes(self):
        self.manual_queue.put(self.current_interpretation)
        self.output_text.insert(tk.END, "Interpretation stored in queue.\n")
        self.clear_yes_no_buttons()

    def store_no(self):
        self.output_text.insert(tk.END, "Interpretation not stored.\n")
        self.clear_yes_no_buttons()

    def clear_yes_no_buttons(self):
        self.yes_no_frame.pack_forget()
        self.yes_no_frame.destroy()
        delattr(self, 'yes_no_frame')

    def view_queue(self):
        queue_list = list(self.manual_queue.queue)
        self.output_text.insert(tk.END, "Queue: " + ", ".join(queue_list) + "\n")

    def speak_queue(self):
        while not self.manual_queue.empty():
            enqueue_tts(self.manual_queue.get())

    def done_adding(self):
        self.output_text.insert(tk.END, "Finished adding files to the queue.\n")

    def append_log(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def update_text_size(self, size):
        self.manual_label.config(font=("Helvetica", size))
        self.view_queue_button.config(font=("Helvetica", size))
        self.speak_queue_button.config(font=("Helvetica", size))
        self.done_button.config(font=("Helvetica", size))
        self.back_button.config(font=("Helvetica", size))
        for widget in self.scrollable_frame.winfo_children():
            widget.config(font=("Helvetica", size))

    def update_window_size(self, scale_factor):
        self.manual_label.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.view_queue_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.speak_queue_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.done_button.config(font=(("Helvetica", int(self.controller.text_size * scale_factor))))
        self.back_button.config(font=(("Helvetica", int(self.controller.text_size * scale_factor))))
        for widget in self.scrollable_frame.winfo_children():
            widget.config(font=(("Helvetica", int(self.controller.text_size * scale_factor))))

    def __del__(self):
        stop_tts_thread()
        self.tts_thread.join()

#///////////////////////////////////////////////////////////////////////
class AutomatedPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()
        self.scanning = False
        self.stop_event = threading.Event()

    def create_widgets(self):
        self.automated_label = tk.Label(self, text="Automated File Location", font=("Helvetica", self.controller.text_size))
        self.automated_label.pack(pady=10)

        self.status_label = tk.Label(self, text="", font=("Helvetica", self.controller.text_size))
        self.status_label.pack(pady=10)

        self.loading_icon = tk.Label(self, text="", width=10, height=2, font=("Helvetica", self.controller.text_size))
        self.loading_icon.pack(pady=5)

        self.output_text = tk.Text(self, height=15, width=70, font=("Helvetica", self.controller.text_size))
        self.output_text.pack(pady=5)
        self.output_text.pack_forget()

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=5)

        self.start_button = tk.Button(self.button_frame, text="Start Scan", command=self.start_automated_mode, font=("Helvetica", self.controller.text_size))
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = tk.Button(self.button_frame, text="Stop Scan", command=self.stop_automated_mode, font=("Helvetica", self.controller.text_size))
        self.stop_button.grid(row=0, column=1, padx=5)

        self.show_logs_button = tk.Button(self.button_frame, text="Show Logs", command=self.toggle_logs, font=("Helvetica", self.controller.text_size))
        self.show_logs_button.grid(row=0, column=2, padx=5)

        self.back_button = tk.Button(self.button_frame, text="Back", command=self.stop_and_back, font=("Helvetica", self.controller.text_size))
        self.back_button.grid(row=0, column=3, padx=5)

    def start_automated_mode(self):
        if not self.scanning:
            self.output_text.delete(1.0, tk.END)
            self.set_status("Scanning . . .")
            self.set_loading_icon("⏳")
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self.monitor_directory)
            self.monitor_thread.start()
            self.scanning = True

    def stop_automated_mode(self):
        if self.scanning:
            self.stop_event.set()
            self.set_status("Scanning paused.")
            self.set_loading_icon("")
            self.scanning = False

    def stop_and_back(self):
        self.stop_automated_mode()
        self.controller.show_frame("MainPage")

    def toggle_logs(self):
        if self.output_text.winfo_ismapped():
            self.output_text.pack_forget()
            self.show_logs_button.config(text="Show Logs")
        else:
            self.output_text.pack(pady=5)
            self.show_logs_button.config(text="Hide Logs")

    def set_status(self, status):
        self.status_label.config(text=status)

    def set_loading_icon(self, icon):
        self.loading_icon.config(text=icon)

    def monitor_directory(self):
        log_queue.put("Starting directory monitoring...")
        processed_files = set(f for f in os.listdir(downloads_dir) if f.endswith('.edf'))
        tts_thread = start_tts_thread()

        while not self.stop_event.is_set():
            new_files = set(f for f in os.listdir(downloads_dir) if f.endswith('.edf')) - processed_files

            if new_files:
                for file_name in new_files:
                    file_path = os.path.join(downloads_dir, file_name)
                    log_queue.put(f"Files detected: {file_name}. Reading . . .")
                    interpretation = classify_edf(file_path)
                    log_queue.put(f".edf File detected: {file_name}, Interpretation: {interpretation}\n")
                    enqueue_tts(interpretation)  # Only speak the interpretation
                    processed_files.add(file_name)
                log_queue.put("Resuming scan . . .")
            else:
                log_queue.put("No new .edf files detected, retrying...\n")
            self.stop_event.wait(3)  # Sleep for 3 seconds

        stop_tts_thread()
        tts_thread.join()
        log_queue.put("Stopped directory monitoring.")

    def append_log(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def update_text_size(self, size):
        self.automated_label.config(font=("Helvetica", size))
        self.status_label.config(font=("Helvetica", size))
        self.loading_icon.config(font=("Helvetica", size))
        self.output_text.config(font=("Helvetica", size))
        self.start_button.config(font=("Helvetica", size))
        self.stop_button.config(font=("Helvetica", size))
        self.show_logs_button.config(font=("Helvetica", size))
        self.back_button.config(font=("Helvetica", size))

    def update_window_size(self, scale_factor):
        self.automated_label.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.status_label.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.loading_icon.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.output_text.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.start_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.stop_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.show_logs_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.back_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))

#///////////////////////////////////////////////////////////////////////
class DebugPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()
        self.current_interpretation = None  # To store the current interpretation

    def create_widgets(self):
        self.debug_label = tk.Label(self, text="Debug File Location", font=("Helvetica", self.controller.text_size))
        self.debug_label.pack(pady=10)

        self.files_frame = tk.Frame(self)
        self.files_frame.pack(pady=5)

        self.files_canvas = tk.Canvas(self.files_frame, height=200)
        self.files_canvas.pack(side=tk.LEFT, fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self.files_frame, orient="vertical", command=self.files_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        self.scrollable_frame = tk.Frame(self.files_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.files_canvas.configure(
                scrollregion=self.files_canvas.bbox("all")
            )
        )

        self.files_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.files_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.load_files()

        self.output_text = tk.Text(self, height=10, width=70, font=("Helvetica", self.controller.text_size))
        self.output_text.pack(pady=5)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=5)

        self.new_button = tk.Button(self.button_frame, text="New", command=self.new_sample, font=("Helvetica", self.controller.text_size))
        self.new_button.grid(row=0, column=0, padx=5)

        self.add_button = tk.Button(self.button_frame, text="Add", command=self.add_sample, font=("Helvetica", self.controller.text_size))
        self.add_button.grid(row=0, column=1, padx=5)

        self.remove_button = tk.Button(self.button_frame, text="Remove", command=self.remove_sample, font=("Helvetica", self.controller.text_size))
        self.remove_button.grid(row=0, column=2, padx=5)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_model, font=("Helvetica", self.controller.text_size))
        self.reset_button.grid(row=0, column=3, padx=5)

        self.all_button = tk.Button(self.button_frame, text="All", command=self.process_all_files, font=("Helvetica", self.controller.text_size))
        self.all_button.grid(row=0, column=4, padx=5)

        self.back_button = tk.Button(self.button_frame, text="Back", command=lambda: self.controller.show_frame("MainPage"), font=("Helvetica", self.controller.text_size))
        self.back_button.grid(row=0, column=5, padx=5)

    def load_files(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for file_name in os.listdir(samples_dir):
            if file_name.endswith('.edf'):
                file_button = tk.Button(self.scrollable_frame, text=file_name, command=lambda fn=file_name: self.process_file(fn), font=("Helvetica", self.controller.text_size))
                file_button.pack(pady=2)

    def process_file(self, file_name):
        file_path = os.path.join(samples_dir, file_name)
        if os.path.exists(file_path):
            start_time = time.time()
            response = classify_edf(file_path)
            end_time = time.time()
            processing_time = end_time - start_time
            self.output_text.insert(tk.END, f"File: {file_name}, Classification: {response}\nTime taken: {processing_time:.2f} seconds\n")
            self.ask_if_correct(response, file_path)
        else:
            self.output_text.insert(tk.END, f"File not found: {file_name}\n")

    def process_all_files(self):
        self.output_text.insert(tk.END, "Processing all files...\n")
        for file_name in os.listdir(samples_dir):
            if file_name.endswith('.edf'):
                file_path = os.path.join(samples_dir, file_name)
                if os.path.exists(file_path):
                    start_time = time.time()
                    response = classify_edf(file_path)
                    end_time = time.time()
                    processing_time = end_time - start_time
                    self.output_text.insert(tk.END, f"File: {file_name}, Classification: {response}\nTime taken: {processing_time:.2f} seconds\n")
        self.output_text.insert(tk.END, "Finished processing all files.\n")

    def ask_if_correct(self, response, file_path):
        self.current_interpretation = response
        self.current_file_path = file_path
        self.output_text.insert(tk.END, "Is this interpretation correct?\n")

        self.yes_no_frame = tk.Frame(self)
        self.yes_no_frame.pack(pady=5)

        self.yes_button = tk.Button(self.yes_no_frame, text="Yes", command=self.correct_interpretation, font=("Helvetica", self.controller.text_size))
        self.yes_button.grid(row=0, column=0, padx=5)

        self.no_button = tk.Button(self.yes_no_frame, text="No", command=self.incorrect_interpretation, font=("Helvetica", self.controller.text_size))
        self.no_button.grid(row=0, column=1, padx=5)

    def correct_interpretation(self):
        retrain_model_with_new_data(load_edf_features(self.current_file_path), self.current_interpretation)
        self.output_text.insert(tk.END, "Interpretation confirmed and model retrained.\nModel trained and saved successfully.\n")
        self.clear_yes_no_buttons()

    def incorrect_interpretation(self):
        self.output_text.insert(tk.END, "Please select the correct interpretation.\n")
        self.clear_yes_no_buttons()
        self.show_correction_buttons()

    def show_correction_buttons(self):
        self.correction_frame = tk.Frame(self)
        self.correction_frame.pack(pady=5)

        # Add buttons for each possible word (e.g., "yes", "no", etc.)
        for word in ['yes', 'no', 'hello', 'please', 'sorry', 'thanks', 'more', 'less', 'none', 'all']:
            tk.Button(self.correction_frame, text=word, command=lambda w=word: self.correct_word_selected(w), font=("Helvetica", self.controller.text_size)).pack(side=tk.LEFT, padx=5)

    def correct_word_selected(self, word):
        retrain_model_with_new_data(load_edf_features(self.current_file_path), word)
        self.output_text.insert(tk.END, f"Correct interpretation: {word}\nModel trained and saved successfully.\n")
        self.clear_correction_buttons()

    def clear_yes_no_buttons(self):
        self.yes_no_frame.pack_forget()
        self.yes_no_frame.destroy()
        delattr(self, 'yes_no_frame')

    def clear_correction_buttons(self):
        self.correction_frame.pack_forget()
        self.correction_frame.destroy()
        delattr(self, 'correction_frame')

    def new_sample(self):
        self.output_text.insert(tk.END, "Select a .edf file to add as a new sample.\n")
        self.show_file_selection_buttons(self.add_new_sample)

    def add_sample(self):
        self.output_text.insert(tk.END, "Select a .edf file to add to the reference directory.\n")
        self.show_file_selection_buttons(self.copy_to_reference)

    def remove_sample(self):
        self.output_text.insert(tk.END, "Select a .edf file to remove from the reference directory.\n")
        self.show_file_selection_buttons(self.remove_from_reference)

    def reset_model(self):
        self.output_text.insert(tk.END, "Type CONFIRM to reset the model.\n")
        self.confirm_entry = tk.Entry(self, font=("Helvetica", self.controller.text_size))
        self.confirm_entry.pack(pady=5)
        self.confirm_button = tk.Button(self, text="Confirm", command=self.confirm_reset, font=("Helvetica", self.controller.text_size))
        self.confirm_button.pack(pady=5)

    def confirm_reset(self):
        if self.confirm_entry.get().strip().upper() == "CONFIRM":
            reset_model()
            self.output_text.insert(tk.END, "Model has been reset.\n")
        else:
            self.output_text.insert(tk.END, "Reset cancelled.\n")
        self.confirm_entry.pack_forget()
        self.confirm_button.pack_forget()
        self.confirm_entry.destroy()
        self.confirm_button.destroy()

    def show_file_selection_buttons(self, command):
        self.file_selection_frame = tk.Frame(self)
        self.file_selection_frame.pack(pady=5)

        for file_name in os.listdir(samples_dir):
            if file_name.endswith('.edf'):
                tk.Button(self.file_selection_frame, text=file_name, command=lambda fn=file_name: command(fn), font=("Helvetica", self.controller.text_size)).pack(pady=2)

    def add_new_sample(self, file_name):
        add_new_label(file_name)
        self.output_text.insert(tk.END, f"New sample added: {file_name}\n")
        self.file_selection_frame.pack_forget()
        self.file_selection_frame.destroy()

    def copy_to_reference(self, file_name):
        add_to_existing_label(file_name)
        self.output_text.insert(tk.END, f"Sample added to reference: {file_name}\n")
        self.file_selection_frame.pack_forget()
        self.file_selection_frame.destroy()

    def remove_from_reference(self, file_name):
        remove_label(file_name)
        self.output_text.insert(tk.END, f"Sample removed from reference: {file_name}\n")
        self.file_selection_frame.pack_forget()
        self.file_selection_frame.destroy()

    def append_log(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def update_text_size(self, size):
        self.debug_label.config(font=("Helvetica", size))
        self.new_button.config(font=("Helvetica", size))
        self.add_button.config(font=("Helvetica", size))
        self.remove_button.config(font=("Helvetica", size))
        self.reset_button.config(font=("Helvetica", size))
        self.all_button.config(font=("Helvetica", size))
        self.back_button.config(font=("Helvetica", size))
        for widget in self.scrollable_frame.winfo_children():
            widget.config(font=("Helvetica", size))

    def update_window_size(self, scale_factor):
        self.debug_label.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.new_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.add_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.remove_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.reset_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.all_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        self.back_button.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))
        for widget in self.scrollable_frame.winfo_children():
            widget.config(font=("Helvetica", int(self.controller.text_size * scale_factor)))

#///////////////////////////////////////////////////////////////////////
class SettingsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        self.settings_label = tk.Label(self, text="Settings", font=("Helvetica", self.controller.text_size))
        self.settings_label.pack(pady=10)

        self.text_size_label = tk.Label(self, text="Text Size:", font=("Helvetica", self.controller.text_size))
        self.text_size_label.pack(pady=5)
        text_sizes = [str(round(x * 0.5, 1)) for x in range(1, 41)]
        self.text_size_combo = ttk.Combobox(self, values=text_sizes, font=("Helvetica", self.controller.text_size))
        self.text_size_combo.pack(pady=5)
        self.text_size_combo.set(str(self.controller.text_size))

        self.window_size_label = tk.Label(self, text="Window Size:", font=("Helvetica", self.controller.text_size))
        self.window_size_label.pack(pady=5)
        window_sizes = ["960x540", "1280x720", "1920x1080"]
        self.window_size_combo = ttk.Combobox(self, values=window_sizes, font=("Helvetica", self.controller.text_size))
        self.window_size_combo.pack(pady=5)
        self.window_size_combo.set(self.controller.window_size)

        self.tts_volume_label = tk.Label(self, text="TTS Volume (0.0 to 1.0):", font=("Helvetica", self.controller.text_size))
        self.tts_volume_label.pack(pady=5)
        volume_values = [str(round(x * 0.1, 1)) for x in range(11)]
        self.tts_volume_combo = ttk.Combobox(self, values=volume_values, font=("Helvetica", self.controller.text_size))
        self.tts_volume_combo.pack(pady=5)
        self.tts_volume_combo.set(str(self.controller.tts_volume))

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)

        self.save_button = tk.Button(self.button_frame, text="Save", command=self.save_settings, font=("Helvetica", self.controller.text_size))
        self.save_button.grid(row=0, column=0, padx=5)

        self.cancel_button = tk.Button(self.button_frame, text="Cancel", command=self.cancel_settings, font=("Helvetica", self.controller.text_size))
        self.cancel_button.grid(row=0, column=1, padx=5)

        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.controller.quit, font=("Helvetica", self.controller.text_size))
        self.exit_button.grid(row=0, column=2, padx=5)

    def save_settings(self):
        text_size = int(float(self.text_size_combo.get()))
        window_size = self.window_size_combo.get()
        tts_volume = float(self.tts_volume_combo.get())

        self.controller.update_text_size(text_size)
        self.controller.update_window_size(window_size)
        self.controller.update_tts_volume(tts_volume)

        messagebox.showinfo("Settings", "Settings saved successfully!")
        self.controller.show_frame("MainPage")

    def cancel_settings(self):
        self.controller.show_frame("MainPage")

if __name__ == "__main__":
    load_model()
    app = EDFAssistantApp()
    app.mainloop()