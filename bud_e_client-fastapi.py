# Import necessary libraries
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import threading
import pyaudio
import wave
import webbrowser
import pyautogui
import io
import asyncio
import websockets
import requests
import os
import subprocess
import tempfile
import sys
import mimetypes

class BudEClient(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bud-E Voice Assistant")
        self.geometry("300x350")

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

        self.status_label = tk.Label(self, text="Status: Ready")
        self.status_label.pack(pady=10)

        self.websocket = None
        self.connect_to_server()

        # Set up the window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def connect_to_server(self):
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                print("Connected to server successfully")
                self.status_label.config(text="Status: Connected to server")
            else:
                print("Failed to connect to server")
                self.status_label.config(text="Status: Failed to connect")
        except requests.exceptions.RequestException:
            print("Server is not available")
            self.status_label.config(text="Status: Server unavailable")

    async def stream_audio(self):
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            self.websocket = websocket
            while self.is_recording:
                data = self.stream.read(44100)
                await websocket.send(data)
                await asyncio.sleep(0.1)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        self.is_recording = True
        self.frames = []
        self.start_button.config(text="Stop Conversation")
        self.status_label.config(text="Status: Recording...")
        
        def run_async_stream():
            asyncio.run(self.stream_audio())

        threading.Thread(target=run_async_stream, daemon=True).start()

    def stop_recording(self):
        if self.stream:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.start_button.config(text="Start Conversation")
            self.status_label.config(text="Status: Stopped")
            self.websocket = None

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
        webbrowser.open(url)
        return f"Opened website: {url}"

    def send_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "rb") as file:
                files = {"file": (os.path.basename(file_path), file)}
                response = requests.post("http://localhost:8000/send_file", files=files)
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
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            elif os.name == 'posix':  # macOS and Linux
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
    app = BudEClient()
    app.mainloop()  # Start the Tkinter event loop