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
        self.geometry("600x600")

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []

        self.start_button = tk.Button(self, text="Start Conversation", command=self.toggle_recording)
        self.start_button.pack(pady=10)

        self.screenshot_button = tk.Button(self, text="Take Screenshot", command=self.take_screenshot)
        self.screenshot_button.pack(pady=10)

        self.open_website_button = tk.Button(self, text="Open Website", command=self.open_website_dialog)
        self.open_website_button.pack(pady=10)

        self.send_file_button = tk.Button(self, text="Send File", command=self.send_file)
        self.send_file_button.pack(pady=10)

        self.update_config_button = tk.Button(self, text="Update ClientConfig in Server", command=self.update_client_config)
        self.update_config_button.pack(pady=10)

        self.status_label = tk.Label(self, text="Status: Ready")
        self.status_label.pack(pady=10)

        self.config_textbox = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=40, height=10)
        self.config_textbox.pack(pady=10)

        self.client_id = None
        self.connect_to_server()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

        self.audio_queue = Queue()
        self.sentence_queue = Queue()
        self.playback_lock = threading.Lock()

    def connect_to_server(self):
        try:
            response = requests.get("http://localhost:8001/")
            if response.status_code == 200:
                print("Connected to server successfully")
                self.status_label.config(text="Status: Connected to server")
                self.client_id = self.request_client_id()
                if self.client_id:
                    self.update_config_textbox()
            else:
                print("Failed to connect to server")
                self.status_label.config(text="Status: Failed to connect")
        except requests.exceptions.RequestException:
            print("Server is not available")
            self.status_label.config(text="Status: Server unavailable")

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
        config = {
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
        response = requests.post(f"http://localhost:8001/update_client_config/{self.client_id}", json=config)
        if response.status_code == 200:
            messagebox.showinfo("Success", "Client configuration updated successfully")
            self.update_config_textbox()
        else:
            messagebox.showerror("Error", f"Failed to update client configuration: {response.text}")

    def update_config_textbox(self):
        if self.client_id:
            response = requests.get(f"http://localhost:8001/get_client_data/{self.client_id}")
            if response.status_code == 200:
                config = response.json()
                self.config_textbox.delete(1.0, tk.END)
                self.config_textbox.insert(tk.END, json.dumps(config, indent=4))
            else:
                print(f"Failed to get client data: {response.text}")

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
            response = requests.post(f"http://localhost:8001/receive_audio?client_id={self.client_id}", files=files)

        if response.status_code == 200:
            data = response.json()
            first_sentence_audio = data['first_sentence_audio']
            sentences = data['sentences']

            # Play the first sentence audio
            self.play_audio(first_sentence_audio)

            # Queue the rest of the sentences for TTS generation and playback
            for sentence in sentences[1:]:
                self.sentence_queue.put(sentence)

            # Start the TTS generation and playback process for the remaining sentences
            threading.Thread(target=self.process_sentences, daemon=True).start()

            # Measure and display latency
            self.measure_latency(start_time)
        else:
            print(f"Failed to send audio segment: {response.text}")


    def process_sentences(self):
        while not self.sentence_queue.empty():
            sentence = self.sentence_queue.get()
            response = requests.post(f"http://localhost:8001/generate_tts", json={"sentence": sentence})
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
                except Exception as e:
                    print(f"Failed to play audio: {e}")

        threading.Thread(target=play_sound, daemon=True).start()

    def update_conversation_history(self, conversation_history):
        self.config_textbox.delete(1.0, tk.END)
        self.config_textbox.insert(tk.END, json.dumps(conversation_history, indent=4))

    def take_screenshot(self):
        screenshot = pyautogui.screenshot()
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            screenshot.save(file_path)
            messagebox.showinfo("Screenshot", f"Screenshot saved as {file_path}")
        else:
            messagebox.showinfo("Info", "Screenshot save cancelled")

    def open_website_dialog(self):
        url = simpledialog.askstring("Open Website", "Enter the URL:")
        if url:
            self.open_website(url)

    def open_website(self, url):
        response = requests.post(f"http://localhost:8001/open_website?client_id={self.client_id}", json={"url": url})
        if response.status_code == 200:
            messagebox.showinfo("Success", f"Opened website: {url}")
        else:
            messagebox.showerror("Error", f"Failed to open website: {response.text}")

    def send_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "rb") as file:
                files = {"file": (os.path.basename(file_path), file)}
                response = requests.post(f"http://localhost:8001/send_file?client_id={self.client_id}", files=files)
                if response.status_code == 200:
                    content_disposition = response.headers.get('Content-Disposition')
                    filename = content_disposition.split('filename=')[1].strip('"') if content_disposition else 'received_file'

                    self.save_and_open_file(response.content, filename)
                    messagebox.showinfo("File Received", f"File received and opened: {filename}")
                else:
                    messagebox.showerror("Error", f"Failed to send file: {response.text}")
        else:
            messagebox.showinfo("Info", "File selection cancelled")

    def save_and_open_file(self, content, filename):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, filename)

            with open(temp_path, "wb") as temp_file:
                temp_file.write(content)

            self.open_file(temp_path)

    def open_file(self, file_path):
        try:
            if os.name == 'nt':
                os.startfile(file_path)
            elif os.name == 'posix':
                opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                subprocess.call([opener, file_path])
            else:
                print(f"Unsupported operating system: {os.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {str(e)}")

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
