import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, scrolledtext
import threading
import pyaudio
import wave
import webbrowser
import pyautogui
import io
import asyncio
import requests
import os
import subprocess
import tempfile
import sys
import mimetypes
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import uvicorn
import json
import sounddevice as sd
import soundfile as sf
from queue import Queue
import time

def read_systemprompt(file_path):
    try:
        with open(file_path, 'r') as file:
            systemprompt = file.read()
        return systemprompt
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

systemprompt_path = 'systemprompt.txt'
systemprompt = read_systemprompt(systemprompt_path)

app = FastAPI()

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8002)

@app.post("/send_file")
async def send_file(file: UploadFile = File(...)):
    file_content = await file.read()
    mime_type, _ = mimetypes.guess_type(file.filename)
    response = Response(content=file_content, media_type=mime_type)
    response.headers["Content-Disposition"] = f'attachment; filename="{file.filename}"'
    return response

class BudEClient(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bud-E Voice Assistant")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        window_width = int(screen_width * 0.33)
        window_height = int(screen_height * 0.8)
        
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)

        self.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []

        self.start_button = tk.Button(self, text="Start Conversation", command=self.toggle_recording)
        self.start_button.pack(pady=5)

        self.screenshot_button = tk.Button(self, text="Take Screenshot", command=self.take_screenshot)
        self.screenshot_button.pack(pady=5)

        self.open_website_button = tk.Button(self, text="Open Website", command=self.open_website_dialog)
        self.open_website_button.pack(pady=5)

        self.send_file_button = tk.Button(self, text="Send File", command=self.send_file)
        self.send_file_button.pack(pady=5)

        self.update_config_button = tk.Button(self, text="Update ClientConfig in Server", command=self.update_client_config)
        self.update_config_button.pack(pady=5)

        self.clear_history_button = tk.Button(self, text="Clear History", command=self.clear_history)
        self.clear_history_button.pack(pady=5)

        self.load_history_button = tk.Button(self, text="Load History", command=self.load_history)
        self.load_history_button.pack(pady=5)

        self.save_history_button = tk.Button(self, text="Save History", command=self.save_history)
        self.save_history_button.pack(pady=5)

        self.load_config_button = tk.Button(self, text="Load Config", command=self.load_config_file)
        self.load_config_button.pack(pady=5)

        self.save_config_button = tk.Button(self, text="Save Config", command=self.save_config_file)
        self.save_config_button.pack(pady=5)

        self.clear_config_button = tk.Button(self, text="Clear Config", command=self.clear_config)
        self.clear_config_button.pack(pady=5)

        self.status_label = tk.Label(self, text="Status: Ready")
        self.status_label.pack(pady=5)

        self.config_textbox = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=60, height=30)
        self.config_textbox.pack(pady=5, expand=True, fill=tk.BOTH)

        self.client_id = None
        self.config = self.load_config()
        self.conversation_history = self.load_conversation_history()
        self.connect_to_server()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

        self.audio_queue = Queue()
        self.sentence_queue = Queue()
        self.playback_lock = threading.Lock()

        self.after(300000, self.cleanup_tts_files)

    def load_config(self):
        try:
            with open('client_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()

    def save_config(self):
        with open('client_config.json', 'w') as f:
            json.dump(self.config, f)

    def get_default_config(self):
        return {
            'LLM-Config': {
                'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                'temperature': 0.7,
                'top_p': 0.95,
                'max_tokens': 400,
                'frequency_penalty': 1.1,
                'presence_penalty': 1.1
            },
            'TTS-Config': {'voice': 'en-us'},
            'Skills': "bla...",
            'Conversation History': [],
            'Scratchpad': {},
            'System Prompt': 'Initial Prompt'
        }

    def load_conversation_history(self):
        try:
            with open('conversation_history.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_conversation_history(self):
        with open('conversation_history.json', 'w') as f:
            json.dump(self.conversation_history, f)

    def clear_history(self):
        self.conversation_history = []
        self.config['Conversation History'] = []
        self.save_conversation_history()
        self.update_config_textbox()
        messagebox.showinfo("History Cleared", "Conversation history has been cleared.")

    def load_history(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.conversation_history = json.load(f)
                self.config['Conversation History'] = self.conversation_history
                self.update_config_textbox()
                messagebox.showinfo("History Loaded", f"Conversation history loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load history: {str(e)}")

    def save_history(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.conversation_history, f)
                messagebox.showinfo("History Saved", f"Conversation history saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save history: {str(e)}")

    def load_config_file(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.config = json.load(f)
                self.conversation_history = self.config.get('Conversation History', [])
                self.update_config_textbox()
                messagebox.showinfo("Config Loaded", f"Configuration loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {str(e)}")

    def save_config_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                config_text = self.config_textbox.get(1.0, tk.END)
                config_data = json.loads(config_text)
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=4)
                self.config = config_data
                messagebox.showinfo("Config Saved", f"Configuration saved to {file_path}")
            except json.JSONDecodeError as e:
                messagebox.showerror("Error", f"Invalid JSON in config textbox: {str(e)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {str(e)}")

    def clear_config(self):
        self.config = self.get_default_config()
        self.conversation_history = []
        self.update_config_textbox()
        messagebox.showinfo("Config Cleared", "Configuration has been reset to default.")

    def update_config_textbox(self):
        self.config_textbox.delete(1.0, tk.END)
        self.config['Conversation History'] = self.conversation_history
        self.config_textbox.insert(tk.END, json.dumps(self.config, indent=4))

    def connect_to_server(self):
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = requests.get("http://localhost:8001/")
                if response.status_code == 200:
                    print("Connected to server successfully")
                    self.status_label.config(text="Status: Connected to server")
                    self.client_id = self.request_client_id()
                    if self.client_id:
                        self.update_config_textbox()
                    return
                else:
                    print(f"Failed to connect to server (Attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.RequestException:
                print(f"Server is not available (Attempt {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        self.status_label.config(text="Status: Failed to connect to server")
        messagebox.showerror("Connection Error", "Failed to connect to the server. Please check if the server is running and try again.")

    def request_client_id(self):
        response = requests.post("http://localhost:8001/generate_client_id")
        if response.status_code == 200:
            client_id = response.json()["client_id"]
            print(f"Client ID: {client_id}")
            return client_id
        else:
            print("Failed to generate client ID")
            return None

    def update_client_config(self):
        try:
            config_text = self.config_textbox.get(1.0, tk.END)
            config_data = json.loads(config_text)
            response = requests.post(f"http://localhost:8001/update_client_config/{self.client_id}", json=config_data)
            if response.status_code == 200:
                self.config = config_data
                messagebox.showinfo("Success", "Client configuration updated successfully on the server")
            else:
                messagebox.showerror("Error", f"Failed to update client configuration on the server: {response.text}")
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON in config textbox: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update config: {str(e)}")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
        self.is_recording = True
        self.frames = []
        self.start_button.config(text="Stop Conversation")
        self.status_label.config(text="Status: Recording...")

        def run_async_stream():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.process_audio())

        threading.Thread(target=run_async_stream, daemon=True).start()

    def stop_recording(self):
        if self.stream:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.start_button.config(text="Start Conversation")
            self.status_label.config(text="Status: Stopped")

    async def process_audio(self):
        recording = False
        silent_count = 0
        speech_buffer = []
        consecutive_confidence = 0

        while self.is_recording:
            data = self.stream.read(512)
            audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio_float).unsqueeze(0)

            with torch.no_grad():
                confidence = self.model(tensor, 16000).item()
            print(confidence)
            if recording:
                if confidence > 0.9:
                    silent_count = 0
                    speech_buffer.append(data)
                else:
                    silent_count += 1
                    if silent_count >= 15:
                        await self.receive_audio_segment(b''.join(speech_buffer))
                        speech_buffer = []
                        silent_count = 0
                        self.stop_recording()
            elif confidence >= 0.9:
                consecutive_confidence += 1
                if consecutive_confidence >= 2:
                    recording = True
                    speech_buffer.append(data)
            else:
                consecutive_confidence = 0

    async def receive_audio_segment(self, audio_data):
        start_time = time.time()
        filename = 'temp_speech.wav'
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(audio_data)

        with open(filename, 'rb') as audio_file:
            files = {"file": (filename, audio_file)}
            response = requests.post(f"http://localhost:8001/receive_audio?client_id={self.client_id}", files=files, json=self.config)

        if response.status_code == 200:
            data = response.json()
            first_sentence_audio = data['first_sentence_audio']
            sentences = data['sentences']
            self.conversation_history = data['updated_conversation_history']
            self.apply_config_updates(data['config_updates'])
            print(self.conversation_history)
            
            self.save_conversation_history()
            
            self.play_audio(first_sentence_audio)

            for sentence in sentences[1:]:
                self.sentence_queue.put(sentence)

            threading.Thread(target=self.process_sentences, daemon=True).start()

            self.measure_latency(start_time)
        else:
            print(f"Failed to send audio segment: {response.text}")

    def apply_config_updates(self, updates):
        if isinstance(updates, dict):
            self.config.update(updates)
            self.save_config()
            self.update_config_textbox()
        else:
            print(f"Unexpected config update format: {type(updates)}")

    def process_sentences(self):
        while not self.sentence_queue.empty():
            sentence = self.sentence_queue.get()
            response = requests.post("http://localhost:8001/generate_tts", json={"sentence": sentence, "config": self.config})
            if response.status_code == 200:
                tts_filename = response.json()["filename"]
                self.audio_queue.put(tts_filename)
                self.play_next_audio()
            else:
                print(f"Failed to generate TTS for sentence: {response.text}")

    def measure_latency(self, start_time):
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency}")
        self.status_label.config(text=f"Status: Latency {latency:.2f}s")

    def play_next_audio(self):
        if not self.audio_queue.empty():
            audio_file = self.audio_queue.get()
            self.play_audio(audio_file)

    def play_audio(self, audio_file):
        def play_sound():
            with self.playback_lock:
                try:
                    data, fs = sf.read(audio_file, dtype='float32')
                    sd.play(data, fs)
                    sd.wait()
                    self.play_next_audio()
                    
                    response = requests.post("http://localhost:8001/delete_tts_file", 
                                             json={"filename": audio_file, "config": self.config})
                    if response.status_code != 200:
                        print(f"Failed to delete TTS file: {response.text}")
                    else:
                        print(f"Successfully deleted TTS file: {audio_file}")
                except Exception as e:
                    print(f"Failed to play audio: {e}")

        threading.Thread(target=play_sound, daemon=True).start()

    def take_screenshot(self):
        response = requests.get(f"http://localhost:8001/take_screenshot?client_id={self.client_id}", json=self.config)
        if response.status_code == 200:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                messagebox.showinfo("Screenshot", f"Screenshot saved as {file_path}")
            else:
                messagebox.showinfo("Info", "Screenshot save cancelled")
        else:
            messagebox.showerror("Error", f"Failed to take screenshot: {response.text}")

    def open_website_dialog(self):
        url = simpledialog.askstring("Open Website", "Enter the URL:")
        if url:
            self.open_website(url)

    def open_website(self, url):
        response = requests.post(f"http://localhost:8001/open_website?client_id={self.client_id}&url={url}", json=self.config)
        if response.status_code == 200:
            messagebox.showinfo("Success", f"Opened website: {url}")
        else:
            messagebox.showerror("Error", f"Failed to open website: {response.text}")

    def send_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "rb") as file:
                files = {"file": (os.path.basename(file_path), file)}
                response = requests.get(f"http://localhost:8001/send_file?client_id={self.client_id}&file={file_path}", 
                                        files=files, json=self.config)
                if response.status_code == 200:
                    save_path = filedialog.asksaveasfilename(defaultextension=os.path.splitext(file_path)[1],
                                                             initialfile=os.path.basename(file_path))
                    if save_path:
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        messagebox.showinfo("File Received", f"File saved as: {save_path}")
                    else:
                        messagebox.showinfo("Info", "File save cancelled")
                else:
                    messagebox.showerror("Error", f"Failed to send file: {response.text}")
        else:
            messagebox.showinfo("Info", "File selection cancelled")

    def cleanup_tts_files(self):
        tts_files = [f for f in os.listdir() if f.startswith("tts_output_") and f.endswith(".wav")]
        for file in tts_files:
            try:
                response = requests.post("http://localhost:8001/delete_tts_file", 
                                         json={"filename": file, "config": self.config})
                if response.status_code == 200:
                    print(f"Cleaned up TTS file: {file}")
                else:
                    print(f"Failed to clean up TTS file {file}: {response.text}")
            except Exception as e:
                print(f"Error during cleanup of TTS file {file}: {e}")

        self.after(300000, self.cleanup_tts_files)  # Run every 5 minutes

    def on_closing(self):
        if self.is_recording:
            self.stop_recording()
        self.p.terminate()
        self.destroy()

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    app = BudEClient()
    app.mainloop()
