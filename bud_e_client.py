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
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.inConversation = False
        self.is_playing_tts = False

        self.start_button = tk.Button(self, text="Start Conversation", command=self.toggle_recording)
        self.start_button.pack(pady=5)

        self.conversation_mode_button = tk.Button(self, text="Start Conversation Mode", command=self.toggle_conversation_mode)
        self.conversation_mode_button.pack(pady=5)

        self.screenshot_button = tk.Button(self, text="Take Screenshot", command=self.take_screenshot)
        self.screenshot_button.pack(pady=5)

        self.open_website_button = tk.Button(self, text="Open Website", command=self.open_website_dialog)
        self.open_website_button.pack(pady=5)

        self.send_file_button = tk.Button(self, text="Send File", command=self.send_file)
        self.send_file_button.pack(pady=5)

        self.update_config_button = tk.Button(self, text="Update ClientConfig in Server", command=self.update_client_config)
        self.update_config_button.pack(pady=5)

        self.stop_playback_button = tk.Button(self, text="Stop Playback", command=self.stop_playback)
        self.stop_playback_button.pack(pady=5)

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
        self.config_textbox.bind("<KeyRelease>", self.on_config_change)

        self.config_status_label = tk.Label(self, text="Config Status: OK")
        self.config_status_label.pack(pady=5)

        self.client_id = None
        self.config = self.load_config()
        self.conversation_history = self.config.get('Conversation History', [])
        self.connect_to_server()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

        self.audio_queue = Queue()
        self.sentence_queue = Queue()
        self.playback_lock = threading.Lock()

        self.after(300000, self.cleanup_tts_files)
        self.after(100, self.check_conversation_mode)

    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()

    def save_config(self):
        with open('config.json', 'w') as f:
            json.dump(self.config, f)

    def get_default_config(self):
        return {
            'LLM-Config': {
                'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                'temperature': 0.7,
                'top_p': 0.95,
                'max_tokens': 400,
                'frequency_penalty': 1.05,
                'presence_penalty': 1.05
            },
            'TTS-Config': {'voice': 'en-us'},
            'Skills': ['edit', 'completion'],
            'Conversation History': [],
            'Scratchpad': {},
            'System Prompt': systemprompt
        }

    def on_config_change(self, event):
        try:
            config_text = self.config_textbox.get(1.0, tk.END)
            new_config = json.loads(config_text)
            self.config = new_config
            self.config_status_label.config(text="Config Status: OK", fg="green")
        except json.JSONDecodeError:
            self.config_status_label.config(text="Config Status: Invalid JSON", fg="red")

    def toggle_conversation_mode(self):
        self.inConversation = not self.inConversation
        if self.inConversation:
            self.conversation_mode_button.config(text="End Conversation Mode")
            self.update_client_config()
            self.start_recording()
        else:
            self.conversation_mode_button.config(text="Start Conversation Mode")
            self.stop_recording()

    def check_conversation_mode(self):
        if self.inConversation and not self.is_recording and self.audio_queue.empty():
            self.start_recording()
        self.after(100, self.check_conversation_mode)

    def update_client_config(self):
        try:
            config_text = self.config_textbox.get(1.0, tk.END)
            config_data = json.loads(config_text)
            response = requests.post(f"http://213.173.96.19:8001/update_client_config/{self.client_id}", json=config_data)
            if response.status_code == 200:
                self.config = config_data
                #messagebox.showinfo("Success", "Client configuration updated successfully on the server")
            else:
                messagebox.showerror("Error", f"Failed to update client configuration on the server: {response.text}")
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON in config textbox: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update config: {str(e)}")

    def toggle_recording(self):
        if not self.is_recording:
            self.update_client_config()
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
            if self.is_playing_tts:
                await asyncio.sleep(0.1)  # Pause processing while TTS is playing
                continue

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
                        if not self.inConversation:
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
            response = requests.post(f"http://213.173.96.19:8001/receive_audio?client_id={self.client_id}", files=files, json=self.config)

        if response.status_code == 200:
            data = response.json()
            
            sentences = data['sentences']
            self.conversation_history = data['updated_conversation_history']
            self.apply_config_updates(data['config_updates'])
            
            self.save_config()
            
            # Process the first sentence immediately
            if sentences:
                first_sentence = sentences[0]
                self.process_sentence(first_sentence)

            # Queue the remaining sentences
            for sentence in sentences[1:]:
                self.sentence_queue.put(sentence)

            threading.Thread(target=self.process_sentences, daemon=True).start()
            self.measure_latency(start_time)
        else:
            print(f"Failed to send audio segment: {response.text}")

    def process_sentences(self):
        while not self.sentence_queue.empty():
            sentence = self.sentence_queue.get()
            self.process_sentence(sentence)

    def process_sentence(self, sentence):
        logging.info(f"Processing sentence: {sentence}")
        response = requests.post("http://213.173.96.19:8001/generate_tts", 
                                 json={"sentence": sentence})
        if response.status_code == 200:
            logging.info("Received TTS response successfully")
            audio_data = response.content
            sample_rate = int(response.headers.get("X-Sample-Rate", 24000))
            channels = int(response.headers.get("X-Channels", 1))
            sample_width = int(response.headers.get("X-Sample-Width", 2))

            # Create a BytesIO object from the audio data
            audio_buffer = io.BytesIO(audio_data)

            # Play the audio directly from the buffer
            self.play_audio(audio_buffer, sample_rate, channels, sample_width)
        else:
            logging.error(f"Failed to generate TTS for sentence: {response.text}")

    def play_audio(self, audio_buffer, sample_rate, channels, sample_width):
        def play_sound():
            with self.playback_lock:
                try:
                    logging.info("Attempting to play audio")
                    self.is_playing_tts = True
                    
                    # Read the audio data from the buffer
                    audio_buffer.seek(0)
                    with wave.open(audio_buffer, 'rb') as wf:
                        data = wf.readframes(wf.getnframes())
                    
                    # Convert the audio data to a numpy array
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]

                    # Play the audio
                    sd.play(audio_data, sample_rate)
                    sd.wait()
                    logging.info("Audio playback completed")

                    self.is_playing_tts = False
                    self.play_next_audio()
                    
                except Exception as e:
                    logging.error(f"Failed to play audio: {str(e)}", exc_info=True)
                finally:
                    self.is_playing_tts = False

        threading.Thread(target=play_sound, daemon=True).start()

    def apply_config_updates(self, updates):
        if isinstance(updates, dict):
            self.config.update(updates)
            self.save_config()
            self.update_config_textbox()
        else:
            print(f"Unexpected config update format: {type(updates)}")

    def update_config_textbox(self):
        self.config_textbox.delete(1.0, tk.END)
        self.config['Conversation History'] = self.conversation_history
        self.config_textbox.insert(tk.END, json.dumps(self.config, indent=4))

    def measure_latency(self, start_time):
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency}")
        self.status_label.config(text=f"Status: Latency {latency:.2f}s")

    def cleanup_tts_files(self):
        temp_dir = tempfile.gettempdir()
        logging.info(f"Cleaning up TTS files in directory: {temp_dir}")
        tts_files = [f for f in os.listdir(temp_dir) if f.endswith(".wav")]
        for file in tts_files:
            try:
                file_path = os.path.join(temp_dir, file)
                os.remove(file_path)
                logging.info(f"Cleaned up TTS file: {file_path}")
            except Exception as e:
                logging.error(f"Error during cleanup of TTS file {file}: {str(e)}")

        self.after(300000, self.cleanup_tts_files)  # Run every 5 minutes

    def play_next_audio(self):
        if not self.audio_queue.empty():
            audio_file = self.audio_queue.get()
            self.play_audio(audio_file)

    def stop_playback(self):
        sd.stop()
        with self.playback_lock:
            while not self.audio_queue.empty():
                self.audio_queue.get()
        self.is_playing_tts = False
        logging.info("Playback stopped and audio queue cleared")

    def take_screenshot(self):
        response = requests.get(f"http://213.173.96.19:8001/take_screenshot?client_id={self.client_id}", json=self.config)
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
        response = requests.post(f"http://213.173.96.19:8001/open_website?client_id={self.client_id}&url={url}", json=self.config)
        if response.status_code == 200:
            messagebox.showinfo("Success", f"Opened website: {url}")
        else:
            messagebox.showerror("Error", f"Failed to open website: {response.text}")

    def send_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "rb") as file:
                files = {"file": (os.path.basename(file_path), file)}
                response = requests.get(f"http://213.173.96.19:8001/send_file?client_id={self.client_id}&file={file_path}", 
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

    def connect_to_server(self):
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = requests.get("http://213.173.96.19:8001/")
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
        config = {
            "LLM_Config": self.config['LLM-Config'],
            "TTS_Config": self.config['TTS-Config'],
            "Skills": self.config['Skills'],
            "Conversation_History": self.conversation_history,
            "Scratchpad": self.config['Scratchpad'],
            "System_Prompt": self.config['System Prompt']
        }
        response = requests.post("http://213.173.96.19:8001/generate_client_id", json=config)
        if response.status_code == 200:
            client_id = response.json()["client_id"]
            print(f"Client ID: {client_id}")
            return client_id
        else:
            print(f"Failed to generate client ID. Status code: {response.status_code}, Response: {response.text}")
            return None

    def on_closing(self):
        if self.is_recording:
            self.stop_recording()
        self.stop_playback()
        self.p.terminate()
        self.destroy()

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    app = BudEClient()
    app.mainloop()
