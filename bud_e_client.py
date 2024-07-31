import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import pyaudio
import wave
import webbrowser
import pyautogui
import io
from flask import Flask, request, send_file

class BudEClient(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bud-E Voice Assistant")
        self.geometry("300x300")

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

        self.status_label = tk.Label(self, text="Status: Ready")
        self.status_label.pack(pady=10)

        # Flask server setup
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/get_audio', methods=['GET'])
        def get_audio():
            audio_stream = self.get_audio_stream()
            if audio_stream:
                return send_file(audio_stream, mimetype="audio/wav", as_attachment=True, download_name="recorded_audio.wav")
            return "No audio data available", 404

        @self.app.route('/take_screenshot', methods=['GET'])
        def take_screenshot():
            screenshot = self.take_screenshot(save=False)
            return send_file(screenshot, mimetype="image/png", as_attachment=True, download_name="screenshot.png")

        @self.app.route('/open_website', methods=['POST'])
        def open_website():
            url = request.json.get('url')
            if url:
                return self.open_website(url)
            return "No URL provided", 400

    def run_flask(self):
        self.app.run(host='0.0.0.0', port=5000)

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
        threading.Thread(target=self.record, daemon=True).start()

    def record(self):
        while self.is_recording:
            data = self.stream.read(1024, exception_on_overflow=False)
            self.frames.append(data)

    def stop_recording(self):
        if self.stream:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            self.start_button.config(text="Start Conversation")
            self.status_label.config(text="Status: Stopped")
            self.save_audio()

    def save_audio(self):
        if not self.frames:
            messagebox.showwarning("Warning", "No audio data to save.")
            self.status_label.config(text="Status: Ready")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if file_path:
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(self.frames))
            messagebox.showinfo("Success", f"Audio saved as {file_path}")
        else:
            messagebox.showinfo("Info", "Audio save cancelled")
        self.status_label.config(text="Status: Ready")

    def get_audio_stream(self):
        if self.frames:
            buffer = io.BytesIO()
            wf = wave.open(buffer, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            buffer.seek(0)
            return buffer
        return None

    def take_screenshot(self, save=True):
        screenshot = pyautogui.screenshot()
        if save:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                screenshot.save(file_path)
                messagebox.showinfo("Screenshot", f"Screenshot saved as {file_path}")
            else:
                messagebox.showinfo("Info", "Screenshot save cancelled")
        else:
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            buffer.seek(0)
            return buffer

    def open_website_dialog(self):
        url = tk.simpledialog.askstring("Open Website", "Enter the URL:")
        if url:
            self.open_website(url)

    def open_website(self, url):
        webbrowser.open(url)
        return f"Opened website: {url}"

    def on_closing(self):
        if self.is_recording:
            self.stop_recording()
        self.p.terminate()
        self.destroy()

if __name__ == "__main__":
    client = BudEClient()
    flask_thread = threading.Thread(target=client.run_flask, daemon=True)
    flask_thread.start()
    client.protocol("WM_DELETE_WINDOW", client.on_closing)
    client.mainloop()