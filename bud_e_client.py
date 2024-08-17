import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, scrolledtext
import base64
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
import platform
import pvporcupine
import struct
import numpy as np

PORCUPINE_API_KEY= "FZg4dv9mLgfj9VkDoLWU8xjyl5dT9bEWVzU1AItDkh3v+46Xbd+GLA==" 
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

import re

def parse_lm_activated_skills(skills_code):
    lm_activated_skills = {}
    
    skill_pattern = re.compile(
        r'#\s*LM\s+ACTIVATED\s+SKILL:\s*'
        r'((?:.*\n)*?)'
        r'(?=\s*#\s*LM\s+ACTIVATED\s+SKILL:|\Z)',
        re.IGNORECASE | re.MULTILINE
    )
    
    for match in skill_pattern.finditer(skills_code):
        skill_info = match.group(1).strip()
        
        title_match = re.search(r'SKILL\s+TITLE:\s*(.*)', skill_info, re.IGNORECASE)
        description_match = re.search(r'DESCRIPTION:\s*(.*(?:\n(?!USAGE INSTRUCTIONS:).*)*)', skill_info, re.IGNORECASE | re.DOTALL)
        usage_match = re.search(r'USAGE\s+INSTRUCTIONS:\s*(.*(?:\n.*)*)', skill_info, re.IGNORECASE | re.DOTALL)
        
        if title_match:
            title = title_match.group(1).strip()
            description = description_match.group(1).strip() if description_match else ""
            usage_instructions = usage_match.group(1).strip() if usage_match else ""
            
            function_match = re.search(r'def\s+(\w+)', skills_code[match.end():])
            function_name = function_match.group(1) if function_match else ""
            
            lm_activated_skills[title] = {
                'description': description,
                'usage_instructions': usage_instructions,
                'function_name': function_name
            }
    
    return lm_activated_skills

def get_updated_system_prompt(skills):
    prompt = "You have access to the following functions:\n\n"
    for skill_name, skill_info in skills.items():
        prompt += f"- {skill_name}: {skill_info['description']}\n"
        prompt += f"  Usage Instructions: {skill_info['usage_instructions']}\n\n"
    
    prompt += "To call a function, use the following format:\n"
    prompt += "function_name:(parameter)\n\n"
    prompt += "Use only one parameter per function call. If you need to call multiple functions, use them in separate lines.\n"
    prompt += "The function calls are case-insensitive, but try to match the given function names as closely as possible.\n"
    prompt += "Always use the correct syntax as explained here."
    
    return prompt

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

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.model, self.utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

        self.audio_queue = Queue()
        self.sentence_queue = Queue()
        self.playback_lock = threading.Lock()
        self.playback_complete = threading.Event()
        self.playback_complete.set()  # Initially set to True

        self.after(300000, self.cleanup_tts_files)
        self.after(100, self.check_conversation_mode)

        self.config = self.load_config()
        self.conversation_history = self.config.get('Conversation_History', [])
        self.update_system_prompt()
        self.connect_to_server()

        self.os_type = platform.system()
        self.setup_porcupine()
        self.inactivity_counter = 0


        self.setup_porcupine()
        self.start_wake_word_detection()

    def setup_porcupine(self):
        if self.os_type == "Windows":
            keyword_path = "hey-buddy_en_windows_v3_0_0.ppn"
        else:
            keyword_path = "hey-buddy_en_linux_v3_0_0.ppn"
        
        self.porcupine = pvporcupine.create(PORCUPINE_API_KEY, keyword_paths=[keyword_path], sensitivities=[0.5])

    def start_wake_word_detection(self):
        self.wake_word_thread = threading.Thread(target=self.wake_word_detection_loop, daemon=True)
        self.wake_word_thread.start()

    def wake_word_detection_loop(self):
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

        try:
            while True:
                pcm = audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Wake word 'Hey Buddy' detected!")
                    self.after(0, self.start_conversation_mode)
        finally:
            audio_stream.close()
            pa.terminate()

    def start_conversation_mode(self):
        if not self.inConversation:
            print("Starting conversation mode")
            self.toggle_conversation_mode()

    def toggle_conversation_mode(self):
        self.inConversation = not self.inConversation
        if self.inConversation:
            self.conversation_mode_button.config(text="End Conversation Mode")
            self.update_client_config()
            self.start_recording()
        else:
            self.conversation_mode_button.config(text="Start Conversation Mode")
            self.stop_recording()




    def clear_config(self):
        systemprompt = read_systemprompt(systemprompt_path)
        self.config = self.get_default_config()
        self.conversation_history = []
        self.config['Conversation_History'] = []
        self.update_client_config()

        self.update_config_textbox()
        self.save_config()
        messagebox.showinfo("Config Cleared", "Configuration has been reset to default.")

    def update_system_prompt(self, add_to_history=True):
        skills_code = self.config.get('Skills', '')
        lm_activated_skills = parse_lm_activated_skills(skills_code)
        
        current_prompt = self.config.get('System_Prompt', '')
        
        function_info_pattern = r'You have access to the following functions:[\s\S]*?Always use the correct syntax as explained here.'
        function_info_match = re.search(function_info_pattern, current_prompt, re.DOTALL)
        
        if function_info_match:
            updated_function_info = get_updated_system_prompt(lm_activated_skills)
            updated_prompt = current_prompt[:function_info_match.start()] + updated_function_info + current_prompt[function_info_match.end():]
        else:
            updated_function_info = get_updated_system_prompt(lm_activated_skills)
            updated_prompt = current_prompt + "\n\n" + updated_function_info if current_prompt else updated_function_info
        
        self.config['System_Prompt'] = updated_prompt
        
        if add_to_history:
            if self.conversation_history and self.conversation_history[0]['role'] == 'system':
                self.conversation_history[0]['content'] = updated_prompt
            else:
                self.conversation_history.insert(0, {"role": "system", "content": updated_prompt})
        
        self.save_config()

    def update_config_textbox(self):
        self.config_textbox.delete(1.0, tk.END)
        config_to_display = self.config.copy()
        config_to_display['Conversation_History'] = self.conversation_history
        self.config_textbox.insert(tk.END, json.dumps(config_to_display, indent=4))

    def toggle_recording(self):
        if not self.is_recording:
            self.update_client_config()
            self.start_recording()
        else:
            self.stop_recording()

    def load_config(self):
        try:
            with open('client_config.json', 'r') as f:
                config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError("Config file should contain a JSON object, not a list")
            return config
        except FileNotFoundError:
            return self.get_default_config()
        except json.JSONDecodeError:
            print("Error: Invalid JSON in config file")
            return self.get_default_config()
        except ValueError as e:
            print(f"Error: {str(e)}")
            return self.get_default_config()
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return self.get_default_config()

    def save_config(self):
        with open('client_config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def concatenate_skills(self, directory):
        skills = ""
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                with open(os.path.join(directory, filename), 'r') as file:
                    skills += file.read() + "\n"
        return skills

    def get_default_config(self):
        return {
            'LLM_Config': {
                'model': 'llama-3.1-70b',
                'temperature': 0.6,
                'top_p': 0.95,
                'max_tokens': 400,
                'frequency_penalty': 1.0,
                'presence_penalty': 1.0
            },
            'TTS_Config': {
                'voice': 'Florian',
                'speed': 'normal'
            },
            'Skills': self.concatenate_skills("./skills/"),
            'Conversation_History': [],
            'Scratchpad': {},
            'System_Prompt': systemprompt
        }

    def on_config_change(self, event):
        try:
            config_text = self.config_textbox.get(1.0, tk.END)
            new_config = json.loads(config_text)
            self.config = new_config
            self.config_status_label.config(text="Config Status: OK", fg="green")
        except json.JSONDecodeError:
            self.config_status_label.config(text="Config Status: Invalid JSON", fg="red")


            
    def check_conversation_mode(self):
        if self.inConversation and not self.is_recording and self.audio_queue.empty():
            self.start_recording()
        self.after(100, self.check_conversation_mode)

    def send_request_with_retry(self, method, url, max_retries=3, retry_delay=1, **kwargs):
        for attempt in range(max_retries):
            try:
                if method.lower() == 'get':
                    response = requests.get(url, **kwargs)
                elif method.lower() == 'post':
                    response = requests.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logging.error(f"All retry attempts failed for {url}")
                    raise

    def update_client_config(self):
        try:
            config_data = {
                "LLM_Config": self.config['LLM_Config'],
                "TTS_Config": self.config['TTS_Config'],
                "Skills": self.config['Skills'],
                "Conversation_History": self.conversation_history,
                "Scratchpad": self.config['Scratchpad'],
                "System_Prompt": self.config['System_Prompt']
            }
            logging.info(f"Sending config to server: {json.dumps(config_data, indent=2)}")
            response = self.send_request_with_retry(
                "POST", 
                f"http://213.173.96.19:8001/update_client_config/{self.client_id}", 
                json=config_data
            )
            if response and response.status_code == 200:
                print(config_data)
                logging.info("Client configuration updated successfully on the server")
            else:
                logging.error("Failed to update client configuration on the server")
        except Exception as e:
            logging.error(f"Failed to update config: {str(e)}")
            messagebox.showerror("Error", f"Failed to update config: {str(e)}")

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
        waiting_for_speech = True

        while self.is_recording:
            if self.is_playing_tts:
                await asyncio.sleep(0.05)  # Pause processing while TTS is playing
                continue

            # Wait for all sentences to be played before capturing new audio
            if not self.playback_complete.is_set():
                await asyncio.sleep(0.05)
                continue

            data = self.stream.read(512)
            audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.from_numpy(audio_float).unsqueeze(0)

            with torch.no_grad():
                confidence = self.model(tensor, 16000).item()
            print(f"Confidence: {confidence}, Recording: {recording}, Waiting for speech: {waiting_for_speech}")


            # Wake word detection
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, data)
            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0 and not self.inConversation:
                print("Wake word detected! Starting conversation mode.")
                self.toggle_conversation_mode()
                waiting_for_speech = False
                recording = True
                speech_buffer = [data]
                silent_count = 0
                continue



            if waiting_for_speech:
                if confidence >= 0.9:
                    consecutive_confidence += 1
                    if consecutive_confidence >= 2:
                        print("Speech detected, starting to record")
                        recording = True
                        waiting_for_speech = False
                        speech_buffer = [data]
                        silent_count = 0
                else:
                    consecutive_confidence = 0
            elif recording:
                if confidence > 0.9:
                    silent_count = 0
                    speech_buffer.append(data)
                else:
                    silent_count += 1
                    speech_buffer.append(data)
                    if silent_count >= 25:
                        print("Silence detected, processing speech")
                        self.playback_complete.clear()  # Reset the event before processing
                        await self.send_audio_segment_to_server(b''.join(speech_buffer))
                        speech_buffer = []
                        silent_count = 0
                        recording = False
                        waiting_for_speech = True
                        consecutive_confidence = 0
                        if not self.inConversation:
                            self.stop_recording()


    async def send_audio_segment_to_server(self, audio_data):
        start_time = time.time()
        filename = 'temp_speech.wav'
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(audio_data)

        with open(filename, 'rb') as audio_file:
            files = {"file": (filename, audio_file, "audio/wav")}
            
            form_data = {
                "client_id": self.client_id,
                "client_config": json.dumps(self.config)
            }
            
            response = requests.post(
                "http://213.173.96.19:8001/receive_audio",
                files=files,
                data=form_data
            )

        if response.status_code == 200:
            data = response.json()
            
            sentences = data['sentences']
            self.conversation_history = data['updated_conversation_history']
            self.apply_config_updates(data['config_updates'])
            
            self.save_config()
            
            # Queue all sentences
            for sentence in sentences:
                self.sentence_queue.put(sentence)

            threading.Thread(target=self.process_sentences, daemon=True).start()
            self.measure_latency(start_time)

    def process_sentences(self):
        while not self.sentence_queue.empty():
            sentence = self.sentence_queue.get()
            self.process_sentence(sentence)
        
        # Signal that all sentences have been processed and played
        self.playback_complete.set()

    def process_sentence(self, sentence):
        logging.info(f"Processing sentence: {sentence}")
        
        tts_config = self.config.get('TTS_Config', {})
        voice = tts_config.get('voice', 'Stefanie')
        speed = tts_config.get('speed', 'normal')
        
        data = {
            "sentence": sentence,
            "client_id": self.client_id,
            "voice": voice,
            "speed": speed
        }
        
        response = requests.post("http://213.173.96.19:8001/generate_tts", json=data)
        
        if response.status_code == 200:
            logging.info("Received TTS response successfully")
            audio_data = response.content
            sample_rate = int(response.headers.get("X-Sample-Rate", 24000))
            channels = int(response.headers.get("X-Channels", 1))
            sample_width = int(response.headers.get("X-Sample-Width", 2))

            audio_buffer = io.BytesIO(audio_data)
            self.play_audio(audio_buffer, sample_rate, channels, sample_width)
        else:
            logging.error(f"Failed to generate TTS for sentence: {response.text}")

    def play_audio(self, audio_buffer, sample_rate, channels, sample_width):
        def play_sound():
            with self.playback_lock:
                try:
                    logging.info("Attempting to play audio")
                    self.is_playing_tts = True
                    
                    audio_buffer.seek(0)
                    with wave.open(audio_buffer, 'rb') as wf:
                        actual_channels = wf.getnchannels()
                        actual_sample_width = wf.getsampwidth()
                        actual_sample_rate = wf.getframerate()
                        n_frames = wf.getnframes()
                        frames = wf.readframes(n_frames)

                    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
                    audio_data = np.frombuffer(frames, dtype=dtype_map.get(actual_sample_width, np.int16))
                    
                    if actual_channels == 2:
                        audio_data = audio_data.reshape(-1, 2)

                    audio_data = audio_data.astype(np.float32) / (2**(8 * actual_sample_width - 1))

                    logging.info(f"Playing audio with sample rate: {actual_sample_rate}, channels: {actual_channels}")

                    sd.play(audio_data, actual_sample_rate)
                    sd.wait()
                    logging.info("Audio playback completed")

                    self.is_playing_tts = False
                    self.play_next_audio()
                    
                except Exception as e:
                    logging.error(f"Failed to play audio: {str(e)}", exc_info=True)
                finally:
                    self.is_playing_tts = False

        threading.Thread(target=play_sound, daemon=True).start()

    def play_next_audio(self):
        if not self.audio_queue.empty():
            audio_file = self.audio_queue.get()
            self.play_audio(audio_file)
        elif self.sentence_queue.empty():
            # If both queues are empty, signal that playback is complete
            self.playback_complete.set()

    def apply_config_updates(self, updates):
        if isinstance(updates, dict):
            self.config.update(updates)
            self.save_config()
            self.update_config_textbox()
        else:
            print(f"Unexpected config update format: {type(updates)}")

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

    def stop_playback(self):
        sd.stop()
        with self.playback_lock:
            while not self.audio_queue.empty():
                self.audio_queue.get()
            while not self.sentence_queue.empty():
                self.sentence_queue.get()
        self.is_playing_tts = False
        self.playback_complete.set()
        logging.info("Playback stopped and all queues cleared")

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
        if self.client_id:
            self.update_client_config()

    def request_client_id(self):
        config = {
            "LLM_Config": self.config['LLM_Config'],
            "TTS_Config": self.config['TTS_Config'],
            "Skills": self.config['Skills'],
            "Conversation_History": self.conversation_history,
            "Scratchpad": self.config['Scratchpad'],
            "System_Prompt": self.config['System_Prompt']
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
        self.porcupine.delete()
        self.destroy()


if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    app = BudEClient()
    app.mainloop()
