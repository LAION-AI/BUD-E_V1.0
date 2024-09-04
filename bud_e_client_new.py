import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import threading
import pyaudio
import wave
import io
import re
import asyncio
import requests
import os
import json
import sounddevice as sd
import soundfile as sf
from queue import Queue, Empty
from dataclasses import dataclass
import time
import logging
import numpy as np
import platform
import pvporcupine
import struct
import torch
import subprocess
import sys

from fastapi import FastAPI
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SERVER_IP="http://213.173.96.19:8001"
RECORDING_TIME_AFTER_LAST_VOICE_ACTIVITY_DETECTED = 25  # UNIT: 512 / 16000 Sec = 32 ms
REQUIRED_VOICE_DETECTION_CONFIDENCE_TO_START_OR_STOP_RECORDING = 0.9


import io
import json
import base64
from PIL import Image
import pyperclip
import platform
import win32clipboard
from io import BytesIO

def get_clipboard_content():
    os_type = platform.system()
    
    if os_type == "Windows":
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
                data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)
                img = Image.open(BytesIO(data))
                return img
            elif win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
            elif win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_TEXT):
                return win32clipboard.GetClipboardData(win32clipboard.CF_TEXT).decode('utf-8')
        finally:
            win32clipboard.CloseClipboard()
    else:  # Linux and others
        try:
            from PIL import ImageGrab
            img = ImageGrab.grabclipboard()
            if img:
                return img
            else:
                return pyperclip.paste()
        except ImportError:
            return pyperclip.paste()
    
    return None

def process_image(img):
    if img.size[0] > 1280 or img.size[1] > 720:
        img.thumbnail((1280, 720), Image.LANCZOS)
    
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=50)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    timestamp = int(time.time())
    img.save(f"test_{timestamp}.jpg", "JPEG", quality=50)
    return {
        "type": "image",
        "format": "jpeg",
        "data": img_str,
        "size": img.size
    }




import subprocess
import sys
import io
import logging
import tempfile
import os

def execute_client_code(codesnippet):
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    logger.debug("Entering execute_client_code")

    logger.debug("Creating temporary file for code execution")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(codesnippet)
        temp_file_path = temp_file.name

    logger.debug(f"Temporary file created at: {temp_file_path}")

    try:
        # Execute the code using subprocess
        logger.debug("Executing code using subprocess")
        result = subprocess.run([sys.executable, temp_file_path], 
                                capture_output=True, 
                                text=True, 
                                check=True)
        
        # Capture and store the output
        output = result.stdout

        # Log and print the output
        logger.info("Code execution completed successfully")
        logger.debug("Output of executed code:")
        logger.debug(output)
        print("Output of executed code:")
        print(output)

    except subprocess.CalledProcessError as e:
        # Handle execution errors
        error_output = f"Error executing code:\n{e.output}\n{e.stderr}"
        logger.error(f"Error during code execution: {error_output}")
        print(error_output)

    except Exception as e:
        # Handle unexpected errors
        error_output = f"Unexpected error executing code: {str(e)}"
        logger.exception(f"Unexpected error during code execution: {str(e)}")
        print(error_output)

    finally:
        # Clean up the temporary file
        logger.debug(f"Cleaning up temporary file: {temp_file_path}")
        os.unlink(temp_file_path)

    logger.debug("Exiting execute_client_code")


@dataclass
class SentenceAudio:
    """Dataclass to hold sentence text and corresponding audio data."""
    text: str
    audio_data: io.BytesIO
    sample_rate: int
    channels: int

class AudioPlayer:
    def __init__(self, playback_completed_callback):
        self.audio_queue = Queue()
        self.is_playing = False
        self.stop_event = threading.Event()
        self.playback_completed_callback = playback_completed_callback
        self.current_stream = None
        self.current_data = None
        self.data_lock = threading.Lock()
        self.complete_stop_flag = False
        self.counter = 0
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def _playback_loop(self):
        while not self.stop_event.is_set():
            try:
                if self.complete_stop_flag:
                    self.clear_queue()
                    self.complete_stop_flag = False
                    continue

                audio_item = self.audio_queue.get(timeout=1)
                if audio_item is None:
                    continue

                self.is_playing = True
                self._play_audio(audio_item)
                self.is_playing = False

            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in audio playback loop: {str(e)}", exc_info=True)

    def _play_audio(self, audio_item):
        if audio_item is None:
            logging.warning("Attempted to play None audio item")
            return

        try:
            print("PLAYING:", self.counter)
            self.counter += 1
            logging.info(f"Playing audio for sentence: {audio_item.text[:30] if audio_item.text else 'No text'}...")
            audio_item.audio_data.seek(0)
            data, sample_rate = sf.read(audio_item.audio_data)
            
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            
            data = np.clip(data, -1.0, 1.0)
            
            if len(data.shape) == 1:
                data = np.column_stack((data, data))
            
            with self.data_lock:
                self.current_data = data

            def audio_callback(outdata, frames, time, status):
                if status:
                    logging.warning(f'Audio callback status: {status}')
                with self.data_lock:
                    if self.current_data is None or len(self.current_data) == 0 or self.complete_stop_flag:
                        raise sd.CallbackStop()
                    elif len(self.current_data) < frames:
                        outdata[:len(self.current_data)] = self.current_data
                        outdata[len(self.current_data):] = 0
                        self.current_data = None
                        raise sd.CallbackStop()
                    else:
                        outdata[:] = self.current_data[:frames]
                        self.current_data = self.current_data[frames:]

            self.current_stream = sd.OutputStream(
                samplerate=sample_rate, channels=data.shape[1],
                callback=audio_callback, finished_callback=self.stream_finished
            )
            self.current_stream.start()
            
            while self.current_stream.active and not self.stop_event.is_set() and not self.complete_stop_flag:
                sd.sleep(100)
            
            if self.current_stream.active:
                self.current_stream.stop()
            self.current_stream.close()
            self.current_stream = None
            
            with self.data_lock:
                self.current_data = None
            
            if not self.stop_event.is_set() and not self.complete_stop_flag:
                logging.info("Audio playback completed")
                if self.playback_completed_callback:
                    self.playback_completed_callback(audio_item.text if audio_item.text else "")
        except Exception as e:
            logging.error(f"Failed to play audio: {str(e)}", exc_info=True)

    def stop_playback(self):
        logging.info("Stopping playback")
        self.stop_event.set()
        self.complete_stop_flag = True
        self.clear_queue()
        if self.current_stream:
            self.current_stream.stop()
        with self.data_lock:
            self.current_data = None
        self.is_playing = False
        self.stop_event.clear()

    def queue_audio(self, audio_item):
        if audio_item is None:
            logging.warning("Attempted to queue None audio item")
            return
        self.complete_stop_flag = False
        self.audio_queue.put(audio_item)

    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break

    def stream_finished(self):
        logging.info("Stream finished callback called")

    def stop(self):
        self.stop_event.set()
        self.clear_queue()
        if self.current_stream:
            self.current_stream.stop()
        with self.data_lock:
            self.current_data = None
        self.playback_thread.join(timeout=1)

class BudEClient(tk.Tk):
    def __init__(self):
        super().__init__()
        self.determine_os()
        self.title("Bud-E Voice Assistant")
        
        
        self.setup_gui()
        self.setup_audio()
        self.load_config()
        self.connect_to_server()  # This will now handle client ID acquisition
        self.setup_wake_word()
        self.inConversation = False
        self.last_wake_word_time = 0
        self.wake_word_cooldown = 2
        #self.setup_fastapi()
  

    def connect_to_server(self):
        if 'API_Key' not in self.config or not self.config['API_Key']:
            messagebox.showerror("API Key Error", "Please set a valid API key in the configuration.")
            return

        try:
            # First, check if the server is accessible
            response = requests.get(SERVER_IP+"/", 
                                    headers={'X-API-Key': str(self.config.get('API_Key', '12345'))})
            if response.status_code == 200:
                self.status_label.config(text="Status: Connected to server")
                
                # Now, request a client ID
                self.request_client_id()
                
                if self.config.get('client_id'):
                    self.update_client_config()  # Update server with full config including client ID

                else:
                    raise Exception("Failed to obtain a valid client ID")
            else:
                raise Exception(f"Server returned status code: {response.status_code}")
        except Exception as e:
            self.status_label.config(text="Status: Failed to connect to server")
            messagebox.showerror("Connection Error", f"Failed to connect to the server: {str(e)}")
    
    
    def determine_os(self):
        self.os_type = "Windows" if platform.system() == "Windows" else "Linux"

    def setup_wake_word(self):
        PORCUPINE_API_KEY = "X+OQINdgT65zWo/BpnunBPRef3uMEWRmES2DjGzwYW/oKQY6kyL9Kw=="
        
        if self.os_type == "Windows":
            hey_buddy_path = "hey-buddy_en_windows_v3_0_0.ppn"
            stop_buddy_path = "stop-buddy_en_windows_v3_0_0.ppn"
        else:
            hey_buddy_path = "hey-buddy_en_linux_v3_0_0.ppn"
            stop_buddy_path = "stop-buddy_en_linux_v3_0_0.ppn"

        try:
            self.porcupine = pvporcupine.create(
                access_key=PORCUPINE_API_KEY,
                keyword_paths=[hey_buddy_path, stop_buddy_path],
                sensitivities=[0.5, 0.5]
            )
            threading.Thread(target=self.wake_word_detection_loop, daemon=True).start()
        except Exception as e:
            logging.error(f"Failed to initialize Porcupine: {str(e)}")
            messagebox.showerror("Porcupine Error", f"Failed to initialize wake word detection: {str(e)}")

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
                current_time = time.time()
                if keyword_index == 0:  # "Hey Buddy" detected
                    if current_time - self.last_wake_word_time > self.wake_word_cooldown:
                        print("Wake word 'Hey Buddy' detected!")
                        self.last_wake_word_time = current_time
                        self.after(0, self.activate_conversation_mode)
                        time.sleep(0.5)
                elif keyword_index == 1:  # "Stop Buddy" detected
                    print("Stop word 'Stop Buddy' detected!")
                    self.after(0, self.deactivate_conversation_mode)
        except Exception as e:
            print(f"Error in wake word detection loop: {e}")
        finally:
            audio_stream.close()
            pa.terminate()

    def activate_conversation_mode(self):
        if not self.inConversation:
            print("Activating conversation mode via wake word")
            self.toggle_recording()

    def deactivate_conversation_mode(self):
        if self.inConversation:
            print("Deactivating conversation mode via stop word")
            self.toggle_recording()
        self.stop_playback()

    def setup_gui(self):
        self.start_button = tk.Button(self, text="Start Conversation", command=self.toggle_recording)
        self.start_button.pack(pady=5)

        self.stop_playback_button = tk.Button(self, text="Stop Playback", command=self.stop_playback)
        self.stop_playback_button.pack(pady=5)

        self.update_config_button = tk.Button(self, text="Update ClientConfig in Server", command=self.update_client_config)
        self.update_config_button.pack(pady=5)

        self.load_config_button = tk.Button(self, text="Load Config", command=self.load_config_file)
        self.load_config_button.pack(pady=5)

        self.save_config_button = tk.Button(self, text="Save Config", command=self.save_config_file)
        self.save_config_button.pack(pady=5)

        self.clear_config_button = tk.Button(self, text="Clear Config", command=self.clear_config)
        self.clear_config_button.pack(pady=5)

        self.conversation_mode_button = tk.Button(self, text="Start Conversation Mode", state=tk.DISABLED)
        self.conversation_mode_button.pack(pady=5)

        self.status_label = tk.Label(self, text="Status: Ready")
        self.status_label.pack(pady=5)

        self.config_textbox = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=60, height=30)
        self.config_textbox.pack(pady=5, expand=True, fill=tk.BOTH)
        self.config_textbox.bind("<KeyRelease>", self.on_config_change)

        self.config_status_label = tk.Label(self, text="Config Status: OK")
        self.config_status_label.pack(pady=5)

    def setup_audio(self):
        self.audio_player = AudioPlayer(self.on_sentence_playback_completed)
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.recording_lock = threading.Lock()
        self.stop_event = threading.Event()

    def load_config(self):
        try:
            with open('client_config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = self.get_default_config()
        self.conversation_history = self.config.get('Conversation_History', [])
        self.update_config_textbox()

    def read_systemprompt(self, file_path):
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
    
    def get_default_config(self):
        return {
            'LLM_Config': {
                'model': 'gpt-4o-mini',
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
            'System_Prompt': self.read_systemprompt('systemprompt.txt'),
            'API_Key': '12345',
            'client_id':self.config['client_id']
        }

    def concatenate_skills(self, directory):
        skills = ""
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                with open(os.path.join(directory, filename), 'r') as file:
                    skills += file.read() + "\n"
        return skills



    def request_client_id(self):
        try:
            response = requests.post(SERVER_IP+"/generate_client_id", 
                                     json=self.config,
                                     headers={'X-API-Key': str(self.config.get('API_Key', '12345'))})
            if response.status_code == 200:
                self.config['client_id'] = response.json()["client_id"]
                self.save_config()  # Save the updated config with the new client ID
                logging.info(f"Obtained client ID: {self.config['client_id']}")
            else:
                raise Exception(f"Failed to obtain client ID. Server returned: {response.status_code}")
        except Exception as e:
            logging.error(f"Error in request_client_id: {str(e)}")
            raise

    def update_client_config(self):
        try:
            response = requests.post(
                f"{SERVER_IP}/update_client_config/{self.config['client_id']}",
                json=self.config,
                headers={'X-API-Key': str(self.config.get('API_Key', '12345'))}
            )
            if response.status_code == 200:
                logging.info("Client configuration updated successfully on the server")
            else:
                logging.error(f"Failed to update client configuration on the server. Status: {response.status_code}")
                messagebox.showerror("Error", "Failed to update client configuration on the server.")
        except Exception as e:
            logging.error(f"Failed to update config: {str(e)}")
            messagebox.showerror("Error", f"Failed to update config: {str(e)}")



    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        with self.recording_lock:
            if self.inConversation:
                return
            try:
                self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
                self.inConversation = True
                self.stop_event.clear()
                self.start_button.config(text="Stop Conversation")
                self.status_label.config(text="Status: Recording...")
                threading.Thread(target=self.process_audio, daemon=True).start()
            except Exception as e:
                logging.error(f"Failed to start recording: {str(e)}")
                messagebox.showerror("Audio Error", "Failed to start recording. Please check your audio device.")

    def stop_recording(self):
        with self.recording_lock:
            if not self.inConversation:
                return
            self.inConversation = False
            self.stop_playback()
            self.stop_event.set()
            if self.stream:
                self.stream.stop_stream()
            self.start_button.config(text="Start Conversation")
            self.status_label.config(text="Status: Stopped")

    def stop_playback(self):
        logging.info("Stopping playback")
        self.audio_player.stop_playback()

        self.status_label.config(text="Status: Playback stopped")
        if self.inConversation:
            self.stop_recording()

    def on_closing(self):
        if hasattr(self, 'inConversation') and self.inConversation:
            self.stop_recording()
        if hasattr(self, 'audio_player'):
            self.audio_player.stop()
        if hasattr(self, 'p'):
            self.p.terminate()
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'ws_thread'):
            self.ws_thread.join(timeout=1)
        self.destroy()

    def process_audio(self):
        logging.info("Starting audio processing thread")
        recording = False
        silent_count = 0
        speech_buffer = []
        consecutive_confidence = 0

        try:
            while not self.stop_event.is_set():
                if self.stream.is_stopped():
                    logging.info("Stream is stopped, exiting process_audio")
                    break

                if self.audio_player.is_playing:
                    time.sleep(0.1)
                    continue

                try:
                    data = self.stream.read(512, exception_on_overflow=False)
                    audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    tensor = torch.from_numpy(audio_float).unsqueeze(0)

                    with torch.no_grad():
                        confidence = self.model(tensor, 16000).item()

                    if not recording:
                        if confidence >= REQUIRED_VOICE_DETECTION_CONFIDENCE_TO_START_OR_STOP_RECORDING:
                            consecutive_confidence += 1
                            if consecutive_confidence >= 2:
                                recording = True
                                speech_buffer = [data]
                                silent_count = 0
                                logging.info("Speech detected, starting recording")
                        else:
                            consecutive_confidence = 0
                    else:
                        if confidence > REQUIRED_VOICE_DETECTION_CONFIDENCE_TO_START_OR_STOP_RECORDING:
                            silent_count = 0
                            speech_buffer.append(data)
                        else:
                            silent_count += 1
                            speech_buffer.append(data)
                            if silent_count >= RECORDING_TIME_AFTER_LAST_VOICE_ACTIVITY_DETECTED :
                                logging.info("Silence detected, processing recorded speech")
                                self.send_audio_segment_to_server(b''.join(speech_buffer))
                                speech_buffer = []
                                silent_count = 0
                                recording = False
                                consecutive_confidence = 0

                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        logging.warning("Audio input overflow occurred. Ignoring this chunk.")
                    else:
                        logging.error(f"IOError in process_audio: {str(e)}", exc_info=True)
                        break
                except Exception as e:
                    logging.error(f"Error in process_audio: {str(e)}", exc_info=True)
                    break

        except Exception as e:
            logging.error(f"Unexpected error in process_audio main loop: {str(e)}", exc_info=True)

        finally:
            logging.info("Exiting process_audio thread")
            with self.recording_lock:
                if self.stream:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                        logging.info("Audio stream closed successfully")
                    except Exception as e:
                        logging.error(f"Error closing stream: {str(e)}", exc_info=True)
                    self.stream = None
                self.is_recording = False
            logging.info("Audio processing thread terminated")

    def retry_operation(self, operation, max_retries=3, delay=0.1):
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay * (2 ** attempt))

    def process_sentence(self, sentence):
        if not sentence or not isinstance(sentence, str):
            logging.warning(f"Invalid sentence received: {sentence}")
            return

        try:
            def tts_operation():
                tts_config = self.config.get('TTS_Config', {})
                data = {
                    "sentence": sentence,
                    "client_id": self.config['client_id'],
                    "voice": tts_config.get('voice', 'Stefanie'),
                    "speed": tts_config.get('speed', 'normal')
                }
                headers = {'X-API-Key': str(self.config.get('API_Key', '12345'))}
                
                response = requests.post(f"{SERVER_IP}/generate_tts", json=data, headers=headers)
                response.raise_for_status()
                return response.content

            audio_content = self.retry_operation(tts_operation)
            
            sentence_audio = SentenceAudio(
                text=sentence,
                audio_data=io.BytesIO(audio_content),
                sample_rate=16000,
                channels=1
            )
            self.audio_player.queue_audio(sentence_audio)
        except Exception as e:
            logging.error(f"Error in process_sentence: {str(e)}", exc_info=True)
            messagebox.showerror("TTS Error", "An error occurred while generating speech. Please try again.")


    def send_audio_segment_to_server(self, audio_data):
        try:
            # Get and process clipboard content
            clipboard_content = get_clipboard_content()
            print("##############################")
            print(clipboard_content)

            if isinstance(clipboard_content, Image.Image):
                clipboard_data = process_image(clipboard_content)
            elif clipboard_content:
                clipboard_data = {
                    "type": "text",
                    "data": clipboard_content
                }
            else:
                clipboard_data = None
            self.config["clipboard"]=json.dumps(clipboard_data)
        
            self.update_client_config()  # Ensure the server has the latest config before sending audio

            def server_operation():
                with io.BytesIO() as wav_file:
                    with wave.open(wav_file, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(16000)
                        wf.writeframes(audio_data)
                    wav_file.seek(0)

                    files = {"file": ("audio.wav", wav_file, "audio/wav")}
                    form_data = {
                        "client_id": self.config['client_id'],
                        "client_config": json.dumps(self.config)
                    }
                    headers = {'X-API-Key': str(self.config.get('API_Key', '12345'))}
                    
                    logging.info(f"Sending request to: {SERVER_IP}/receive_audio")
                    logging.info(f"Headers: {headers}")
                    logging.info(f"Form data: {form_data}")
                    
                    response = requests.post(f"{SERVER_IP}/receive_audio",
                                             files=files,
                                             data=form_data,
                                             headers=headers)
                    
                    logging.info(f"Response status code: {response.status_code}")
                    logging.info(f"Response headers: {response.headers}")
                    logging.info(f"Response content: {response.text[:500]}...")  # Log first 500 chars of response
                    
                    response.raise_for_status()
                    return response.content

            response_content = self.retry_operation(server_operation)


            parts = response_content.split(b'\n---AUDIO_DATA---\n')
            if len(parts) == 2:
                json_data, audio_data = parts
                data = json.loads(json_data.decode('utf-8'))
                
                self.conversation_history = data.get('updated_conversation_history', self.conversation_history)
                config_updates = data.get('config_updates', {})
                if config_updates:
                    self.config.update(config_updates)
                    self.save_config()
                    self.update_config_textbox()
                
                # execute code in self.config["code_for_client_execution"]
                print("***************************")

                sentences = data.get('sentences') or [data.get('response')]
                print("SENTNECES:", sentences)
                print("[data.get('response')]:", [data.get('response')])
                print("sentences[0]:", sentences[0]) 

                if self.config["code_for_client_execution"] and len(self.config["code_for_client_execution"])>0:
                    print(self.config["code_for_client_execution"])

                    execute_client_code(self.config["code_for_client_execution"])
                    self.config["code_for_client_execution"] = ""

                if sentences and isinstance(sentences[0], str):
                    sentence_audio = SentenceAudio(
                        text=sentences[0],
                        audio_data=io.BytesIO(audio_data),
                        sample_rate=16000,
                        channels=1
                    )
                    print(str(SentenceAudio)[:500])
                    self.audio_player.queue_audio(sentence_audio)
                    for sentence in sentences[1:]:     
                        if isinstance(sentence, str):
                            self.process_sentence(sentence)
                        else:
                            logging.warning(f"Invalid sentence received: {sentence}")
            else:
                logging.error("Invalid response format from server")
                messagebox.showerror("Server Error", "Received an invalid response from the server.")

        except ConnectionError as e:
            logging.error(f"WebSocket connection error: {str(e)}")
            messagebox.showerror("Connection Error", "Failed to establish WebSocket connection with the server.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response'):
                logging.error(f"Response status code: {e.response.status_code}")
                logging.error(f"Response headers: {e.response.headers}")
                logging.error(f"Response content: {e.response.text}")
            messagebox.showerror("Server Error", f"Failed to communicate with the server: {str(e)}")
        except Exception as e:
            logging.error(f"Error in send_audio_segment_to_server: {str(e)}", exc_info=True)
            messagebox.showerror("Error", "An error occurred while processing your request.")

    def on_sentence_playback_completed(self, sentence_text):
        logging.info(f"Completed playback of sentence: {sentence_text[:30]}...")

    def load_config_file(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.config = json.load(f)
                self.conversation_history = self.config.get('Conversation_History', [])
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
        self.config['Conversation_History'] = []
        self.update_client_config()
        self.update_config_textbox()
        self.save_config()
        messagebox.showinfo("Config Cleared", "Configuration has been reset to default.")

    def update_config_textbox(self):
        print("UPDATING TEXTBOX")
        self.config_textbox.delete(1.0, tk.END)
        config_to_display = self.config.copy()
        config_to_display['Conversation_History'] = self.conversation_history
        self.config_textbox.insert(tk.END, json.dumps(config_to_display, indent=4))

    def on_config_change(self, event):
        try:
            config_text = self.config_textbox.get(1.0, tk.END)
            new_config = json.loads(config_text)
            self.config = new_config
            self.config_status_label.config(text="Config Status: OK", fg="green")
        except json.JSONDecodeError:
            self.config_status_label.config(text="Config Status: Invalid JSON", fg="red")

    def save_config(self):
        with open('client_config.json', 'w') as f:
            json.dump(self.config, f, indent=4)



if __name__ == "__main__":
    app = BudEClient()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
