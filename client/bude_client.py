# Import standard libraries
import os  # For file and path operations
import sys  # For system-specific parameters and functions
import time  # For time-related functions
import json  # For JSON encoding and decoding
import threading  # For threading operations
from queue import Queue, Empty  # For threading queues
import logging  # For logging messages
import platform  # For detecting the operating system
import struct  # For handling binary data
import tempfile  # For creating temporary files
import subprocess  # For running subprocesses
import base64  # For base64 encoding
from dataclasses import dataclass  # For data classes
from io import BytesIO  # For byte stream operations
import re

# Import third-party libraries
import tkinter as tk  # For GUI
from tkinter import messagebox, filedialog, scrolledtext  # For GUI components
from PIL import Image, ImageTk  # For image handling
import numpy as np  # For numerical operations
import torch  # For machine learning models
import requests  # For HTTP requests
import sounddevice as sd  # For audio playback
import soundfile as sf  # For audio file handling
import pyaudio  # For audio input/output
import wave  # For WAV file handling
import pvporcupine  # For wake-word detection
import pyperclip  # For clipboard operations
# Optional imports for handling clipboard on Windows
if platform.system() == "Windows":
    try:
        import win32clipboard  # For Windows clipboard access
    except ImportError:
        win32clipboard = None
else:
    win32clipboard = None

# Configure logging
# (Unnecessary logging has been commented out as per request)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
SERVER_IP = "http://213.173.96.19:8001"  # Server IP and port
RECORDING_TIME_AFTER_LAST_VOICE_ACTIVITY_DETECTED = 25  # Frames to wait after last voice activity (25 * 32ms frames)
REQUIRED_VOICE_DETECTION_CONFIDENCE_TO_START_OR_STOP_RECORDING = 0.9  # Threshold for voice activity detection

# Default image paths for the assistant's avatar
DEFAULT_IMAGE1_PATH = "bud-e.png"
DEFAULT_IMAGE2_PATH = "bud-e2.png"

def get_clipboard_content():
    """
    Retrieves the content from the clipboard, supporting both text and images,
    and returns it in an appropriate format depending on the operating system.
    """
    os_type = platform.system()  # Get the operating system type

    if os_type == "Windows":
        # For Windows OS
        if win32clipboard is None:
            # win32clipboard couldn't be imported
            return None

        win32clipboard.OpenClipboard()  # Open the clipboard for data access
        try:
            # Check if the clipboard contains an image (DIB format)
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB):
                data = win32clipboard.GetClipboardData(win32clipboard.CF_DIB)  # Get the image data
                img = Image.open(BytesIO(data))  # Create an Image object from the data
                return img
            # Check if the clipboard contains Unicode text
            elif win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                text_data = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)  # Get the text data
                return text_data
            # Check if the clipboard contains ASCII text
            elif win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_TEXT):
                text_data = win32clipboard.GetClipboardData(win32clipboard.CF_TEXT).decode('utf-8')  # Get and decode the text data
                return text_data
        finally:
            win32clipboard.CloseClipboard()  # Always close the clipboard
    else:
        # For Linux and other operating systems
        try:
            from PIL import ImageGrab  # Import ImageGrab for capturing clipboard images
            img = ImageGrab.grabclipboard()  # Try to grab image from clipboard
            if img:
                # If an image is found
                return img
            else:
                # If no image, try to get text data
                return pyperclip.paste()
        except ImportError:
            # If ImageGrab is not available, return text data
            return pyperclip.paste()

    return None  # Return None if no clipboard content is found

def process_image(img):
    """
    Processes an image by resizing it if necessary and encoding it for transmission.
    Returns a dictionary with image data and metadata.

    Parameters:
    - img: PIL.Image object representing the image to be processed.

    Returns:
    - A dictionary containing:
        - 'type': The type of data ('image').
        - 'format': The image format ('jpeg').
        - 'data': The base64-encoded image data.
        - 'size': Tuple of the image size (width, height).
    """
    # Resize the image if it exceeds specified dimensions
    if img.size[0] > 1280 or img.size[1] > 720:
        img.thumbnail((1280, 720), Image.LANCZOS)  # Resize the image, maintaining aspect ratio

    buffered = BytesIO()  # Create a buffer to hold the image data
    img.save(buffered, format="JPEG", quality=50)  # Save the image into the buffer in JPEG format with quality 50
    img_str = base64.b64encode(buffered.getvalue()).decode()  # Encode the image data in base64

    timestamp = int(time.time())  # Get the current timestamp
    img.save(f"test_{timestamp}.jpg", "JPEG", quality=50)  # Optionally save a copy of the image locally with timestamp (for debugging)

    return {
        "type": "image",
        "format": "jpeg",
        "data": img_str,
        "size": img.size
    }

def execute_client_code(codesnippet):
    """
    Executes a given code snippet safely in a temporary file environment.
    Captures and prints the output or errors.

    Parameters:
    - codesnippet: str, the code to be executed.
    """
    # Create a logger for this function
    logger = logging.getLogger(__name__)

    # Create a temporary file to hold the code snippet
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(codesnippet)  # Write the code to the temporary file
        temp_file_path = temp_file.name  # Get the path of the temporary file

    try:
        # Execute the code using subprocess
        result = subprocess.run([sys.executable, temp_file_path],
                                capture_output=True,
                                text=True,
                                check=True)

        output = result.stdout  # Get the standard output from the execution

        # Print the output of the executed code
        print("Output of executed code:")
        print(output)

    except subprocess.CalledProcessError as e:
        # If the subprocess execution fails
        error_output = f"Error executing code:\n{e.output}\n{e.stderr}"
        logger.error(f"Error during code execution: {error_output}")
        print(error_output)

    except Exception as e:
        # Handle unexpected exceptions
        error_output = f"Unexpected error executing code: {str(e)}"
        logger.exception(f"Unexpected error during code execution: {str(e)}")
        print(error_output)

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

@dataclass
class SentenceAudio:
    """
    Dataclass to hold sentence text and corresponding audio data.

    Attributes:
    - text: The text of the sentence.
    - audio_data: A BytesIO object containing the audio data.
    - sample_rate: The sample rate of the audio.
    - channels: The number of audio channels.
    """
    text: str
    audio_data: BytesIO
    sample_rate: int
    channels: int

class AudioPlayer:
    """
    AudioPlayer class handles queuing and playback of audio data in a separate thread.
    It ensures smooth audio playback and allows for stopping and clearing the queue.
    """

    def __init__(self, playback_completed_callback):
        """
        Initialize the AudioPlayer.

        Parameters:
        - playback_completed_callback: Function to call when playback of a sentence is completed.
        """
        # Initialize attributes
        self.audio_queue = Queue()  # Queue to hold SentenceAudio objects for playback
        self.is_playing = False  # Flag to indicate if audio is currently playing
        self.stop_event = threading.Event()  # Event to signal stopping of playback
        self.playback_completed_callback = playback_completed_callback  # Callback after playback completes
        self.current_stream = None  # Current audio stream
        self.current_data = None  # Current audio data being played
        self.data_lock = threading.Lock()  # Lock for thread-safe access to audio data
        self.complete_stop_flag = False  # Flag to indicate complete stop
        self.counter = 0  # Counter for playback (for debugging)

        # Start the playback thread
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def is_audio_active(self):
        """
        Checks if any audio is currently playing or queued.
        """
        return self.is_playing or not self.audio_queue.empty()
        
    def _playback_loop(self):
        """
        The main loop that continuously checks for audio items in the queue and plays them.
        """
        while not self.stop_event.is_set():
            try:
                if self.complete_stop_flag:
                    # If complete stop is requested, clear the queue and reset the flag
                    self.clear_queue()
                    self.complete_stop_flag = False
                    continue

                # Get next audio item from the queue
                audio_item = self.audio_queue.get(timeout=1)
                if audio_item is None:
                    continue

                self.is_playing = True  # Set playing flag
                self._play_audio(audio_item)  # Play the audio item
                self.is_playing = False  # Reset playing flag

            except Empty:
                # No audio item in queue, continue the loop
                continue
            except Exception as e:
                # Log any errors during playback
                logging.error(f"Error in audio playback loop: {str(e)}", exc_info=True)

    def _play_audio(self, audio_item):
        """
        Plays a single audio item using sounddevice.

        Parameters:
        - audio_item: An instance of SentenceAudio containing audio data and metadata.
        """
        if audio_item is None:
            logging.warning("Attempted to play None audio item")
            return

        try:
            # For debugging purposes, we can print the counter
            # print("PLAYING:", self.counter)
            self.counter += 1
            # We can log the sentence being played (commented out for less verbose output)
            # logging.info(f"Playing audio for sentence: {audio_item.text[:30] if audio_item.text else 'No text'}...")

            audio_item.audio_data.seek(0)  # Seek to the beginning of the audio data
            data, sample_rate = sf.read(audio_item.audio_data)  # Read the audio data

            if data.dtype != np.float32:
                # Ensure data type is float32
                data = data.astype(np.float32)

            data = np.clip(data, -1.0, 1.0)  # Clip audio data to valid range

            if len(data.shape) == 1:
                # Convert mono to stereo by duplicating channels
                data = np.column_stack((data, data))

            with self.data_lock:
                self.current_data = data  # Store current data thread-safely

            def audio_callback(outdata, frames, time, status):
                """
                Callback function for the audio stream.
                """
                if status:
                    # We can log any status messages (commented out for less verbose output)
                    # logging.warning(f'Audio callback status: {status}')
                    pass
                with self.data_lock:
                    if self.current_data is None or len(self.current_data) == 0 or self.complete_stop_flag:
                        # If data is exhausted or stop is requested, stop playback
                        raise sd.CallbackStop()
                    elif len(self.current_data) < frames:
                        # If remaining data is less than required frames
                        outdata[:len(self.current_data)] = self.current_data
                        outdata[len(self.current_data):] = 0  # Fill the rest with silence
                        self.current_data = None  # Reset current data
                        raise sd.CallbackStop()
                    else:
                        # Provide the required frames to outdata
                        outdata[:] = self.current_data[:frames]
                        self.current_data = self.current_data[frames:]  # Update current data

            # Create an output stream with the audio callback
            self.current_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=data.shape[1],
                callback=audio_callback,
                finished_callback=self.stream_finished
            )
            self.current_stream.start()  # Start the audio stream

            # Wait while the stream is active, but allow interruption
            while self.current_stream.active and not self.stop_event.is_set() and not self.complete_stop_flag:
                sd.sleep(100)

            if self.current_stream.active:
                self.current_stream.stop()
            self.current_stream.close()
            self.current_stream = None  # Reset current stream

            with self.data_lock:
                self.current_data = None  # Reset current data

            if not self.stop_event.is_set() and not self.complete_stop_flag:
                # Playback completed successfully
                if self.playback_completed_callback:
                    self.playback_completed_callback(audio_item.text if audio_item.text else "")
        except Exception as e:
            logging.error(f"Failed to play audio: {str(e)}", exc_info=True)

    def stop_playback(self):
        """
        Stops audio playback and clears the queue.
        """
        # logging.info("Stopping playback")
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
        """
        Adds an audio item to the queue for playback.

        Parameters:
        - audio_item: An instance of SentenceAudio to be played.
        """
        if audio_item is None:
            logging.warning("Attempted to queue None audio item")
            return
        self.complete_stop_flag = False
        self.audio_queue.put(audio_item)

    def clear_queue(self):
        """
        Clears all pending audio items from the queue.
        """
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break

    def stream_finished(self):
        """
        Callback function called when the audio stream finishes.
        """
        # logging.info("Stream finished callback called")
        pass

    def stop(self):
        """
        Stops the playback thread and cleans up resources.
        """
        self.stop_event.set()
        self.clear_queue()
        if self.current_stream:
            self.current_stream.stop()
        with self.data_lock:
            self.current_data = None
        self.playback_thread.join(timeout=1)

class BudEClient(tk.Tk):
    """
    Main class for the Bud-E Voice Assistant client application.
    Inherits from tk.Tk for GUI functionality.
    """

    def __init__(self):
        """
        Initializes the Bud-E client application, sets up the GUI components,
        audio configurations, wake word detection, and connects to the server.
        """
        super().__init__()  # Initialize the Tkinter parent class
        
        self.config={}

        # Determine the operating system
        self.determine_os()

        # Set up the main window properties
        self.title("Bud-E Voice Assistant")  # Set window title
        self.geometry("300x500")  # Set window size
        self.resizable(False, False)  # Disable window resizing

        # Set up GUI components
        self.setup_main_gui()

        # Set up audio configurations
        self.setup_audio()

        # Load configuration settings
        self.load_config()

        # Initialize server connection variables
        self.server_connected = False
        self.reconnection_thread = None

        # Set up wake word detection
        self.setup_wake_word()

        # Conversation state variables
        self.inConversation = False  # Flag to indicate if in conversation
        self.last_wake_word_time = 0  # Timestamp of the last wake word detection
        self.wake_word_cooldown = 2  # Cooldown period between wake word detections in seconds

        # Initialize animation variables
        self.current_image = 0  # Index of the current image displayed
        self.animation_running = False  # Flag to indicate if animation is running
        self.animation_thread = None  # Thread for the animation
        self.animation_lock = threading.Lock()  # Lock for thread-safe animation
        self.animation_event = threading.Event()  # Event to control animation timing

        # Checking if animation needs to be stopped, if no sound is being played
        self.animation_monitor_thread = threading.Thread(target=self.monitor_animation, daemon=True)
        self.animation_monitor_thread.start()

        # Get the animation interval from config or use default
        self.animation_interval = float(self.config.get('animation_interval', 1.0))

        # Position the window on the screen
        self.position_window()

        # Connect to the server and start reconnection thread
        self.connect_to_server()
        self.start_reconnection_thread()

    def determine_os(self):
        """
        Determines the operating system type and sets it to self.os_type.
        """
        self.os_type = "Windows" if platform.system() == "Windows" else "Linux"

    def setup_main_gui(self):
        """
        Sets up the main GUI components of the application, including the image display,
        conversation button, and advanced settings button.
        """
        # Image display label
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)  # Add padding

        # Load the default image
        self.load_image(DEFAULT_IMAGE1_PATH)

        # Start/Stop Conversation button
        self.conversation_button = tk.Button(self, text="Start Conversation", command=self.toggle_recording, bg="SystemButtonFace")
        self.conversation_button.pack(pady=10)

        # Advanced Settings button
        self.advanced_settings_button = tk.Button(self, text="Advanced Settings", command=self.open_advanced_settings)
        self.advanced_settings_button.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self, text="Status: Ready")
        self.status_label.pack(pady=5)

    def update_conversation_history(self, new_history):
        """
        Central method to update conversation history across all interfaces.
        """
        self.conversation_history = new_history
        self.config['Conversation_History'] = new_history
        self.save_config()
        self.update_config_textbox()
        
        # Update the history window if it's open
        if hasattr(self, 'history_textbox') and self.history_textbox.winfo_exists():
            self.update_history_textbox()

    def position_window(self):
        """
        Positions the main window at the bottom-right corner of the screen.
        """
        # Get screen width and height
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate position for bottom-right corner
        x = screen_width - 300  # 300 is the width of the window
        y = screen_height - 500  # 500 is the height of the window

        # Set the window's position
        self.geometry(f"300x500+{x}+{y}")

    def load_image(self, path):
        """
        Loads an image from the given path and displays it in the image_label.

        Parameters:
        - path: str, the file path to the image.
        """
        try:
            image = Image.open(path)  # Open the image file
            image = image.resize((300, 300), Image.LANCZOS)  # Resize the image
            photo = ImageTk.PhotoImage(image)  # Create a PhotoImage object
            self.image_label.config(image=photo)  # Update the image_label with the new image
            self.image_label.image = photo  # Keep a reference to avoid garbage collection
        except Exception as e:
            logging.error(f"Failed to load image: {str(e)}")
            messagebox.showerror("Image Error", f"Failed to load image: {str(e)}")

    def open_advanced_settings(self):
        """
        Opens the advanced settings window.
        """
        # Check if the advanced settings window is already open
        if hasattr(self, 'advanced_window') and self.advanced_window.winfo_exists():
            self.advanced_window.lift()  # Bring the window to the front
            return

        # Create a new window for advanced settings
        self.advanced_window = tk.Toplevel(self)
        self.advanced_window.title("Advanced Settings")
        self.setup_advanced_gui(self.advanced_window)  # Set up the advanced GUI components

    def setup_advanced_gui(self, window):
        """
        Sets up the advanced settings GUI components.

        Parameters:
        - window: The Tkinter Toplevel window to place the components in.
        """
        # Start/Stop Conversation button
        self.start_button = tk.Button(window, text="Start Conversation", command=self.toggle_recording, bg="SystemButtonFace")
        self.start_button.pack(pady=5)

        # Stop Playback button
        self.stop_playback_button = tk.Button(window, text="Stop Playback", command=self.stop_playback)
        self.stop_playback_button.pack(pady=5)

        # Update ClientConfig button
        self.update_config_button = tk.Button(window, text="Update ClientConfig in Server", command=self.update_client_config)
        self.update_config_button.pack(pady=5)

        # Load Config button
        self.load_config_button = tk.Button(window, text="Load Config", command=self.load_config_file)
        self.load_config_button.pack(pady=5)

        # Save Config button
        self.save_config_button = tk.Button(window, text="Save Config", command=self.save_config_file)
        self.save_config_button.pack(pady=5)

        # Clear Config button
        self.clear_config_button = tk.Button(window, text="Clear Config", command=self.clear_config)
        self.clear_config_button.pack(pady=5)

        # Select Avatar Image(s) button
        self.select_image_button = tk.Button(window, text="Select Avatar-Image(s)", command=self.select_image)
        self.select_image_button.pack(pady=5)

        # New button: Remove Last Conversation
        self.remove_last_conversation_button = tk.Button(window, text="Remove Last Conversation", command=self.remove_last_conversation)
        self.remove_last_conversation_button.pack(pady=5)
        
        self.view_history_button = tk.Button(window, text="View Conversation History", command=self.open_conversation_history_window)
        self.view_history_button.pack(pady=5)

        # Animation Interval input
        tk.Label(window, text="Animation Interval (seconds):").pack(pady=5)
        self.animation_interval_entry = tk.Entry(window)
        self.animation_interval_entry.insert(0, str(self.animation_interval))
        self.animation_interval_entry.pack(pady=5)
        self.animation_interval_entry.bind("<FocusOut>", self.update_animation_interval)

        # Status label
        self.status_label = tk.Label(window, text="Status: Ready")
        self.status_label.pack(pady=5)

        # Config text box
        self.config_textbox = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=30)
        self.config_textbox.pack(pady=5, expand=True, fill=tk.BOTH)
        self.config_textbox.bind("<KeyRelease>", self.on_config_change)

        # Config status label
        self.config_status_label = tk.Label(window, text="Config Status: OK")
        self.config_status_label.pack(pady=5)
        self.update_config_textbox()

        # New button: View Conversation History
        self.view_history_button = tk.Button(window, text="View Conversation History", command=self.open_conversation_history_window)
        self.view_history_button.pack(pady=5)

        # Use after() to schedule the update after the window is fully created
        window.after(100, self.update_config_textbox)


    def open_conversation_history_window(self):
        """
        Opens a new window with a large text box displaying the conversation history,
        now including a vertical scrollbar.
        """
        history_window = tk.Toplevel(self)
        history_window.title("Conversation History")

        # Calculate window size (90% of screen dimensions)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)

        # Set window size and position
        history_window.geometry(f"{window_width}x{window_height}+{int(screen_width*0.05)}+{int(screen_height*0.05)}")

        # Create a frame to hold the text widget and scrollbar
        text_frame = tk.Frame(history_window)
        text_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Create text widget
        self.history_textbox = tk.Text(text_frame, wrap=tk.WORD, font=("TkDefaultFont", 14))
        self.history_textbox.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Create vertical scrollbar
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.history_textbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the text widget to use the scrollbar
        self.history_textbox.config(yscrollcommand=scrollbar.set)

        # Create a frame for buttons
        button_frame = tk.Frame(history_window)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        # Add Save Changes button
        save_button = tk.Button(button_frame, text="Save Changes", command=self.save_history_changes)
        save_button.pack(side=tk.LEFT, padx=5)

        # Add Discard Changes button
        discard_button = tk.Button(button_frame, text="Discard Changes", command=self.update_history_textbox)
        discard_button.pack(side=tk.LEFT, padx=5)

        # Populate text widget with conversation history
        self.update_history_textbox()
        

    def save_history_changes(self):
        """
        Saves the changes made in the history textbox.
        """
        try:
            new_history = json.loads(self.history_textbox.get(1.0, tk.END))
            self.update_conversation_history(new_history)
            messagebox.showinfo("Success", "Changes saved successfully.")
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON format. Please correct the format and try again.")

    def update_history_textbox(self):
        """
        Updates the conversation history text box with the current history.
        """
        if hasattr(self, 'history_textbox') and self.history_textbox.winfo_exists():
            self.history_textbox.delete(1.0, tk.END)
            history_text = json.dumps(self.conversation_history, indent=2)
            self.history_textbox.insert(tk.END, history_text)


    def on_history_change(self, event):
        """
        Updates the conversation history when changes are made in the history text box.
        """
        try:
            new_history = json.loads(self.history_textbox.get(1.0, tk.END))
            self.update_conversation_history(new_history)
            self.save_config()
            self.update_config_textbox()
        except json.JSONDecodeError:
            # If the JSON is invalid, don't update
            pass


    def remove_last_conversation(self):
        """
        Removes the last element from the conversation history,
        displays a message box, and shows the first 500 characters of the removed element.
        """
        if self.conversation_history:
            removed_conversation = self.conversation_history.pop()
            self.update_conversation_history(self.conversation_history)

            # Display success message
            messagebox.showinfo("Success", "Last conversation removed successfully.")

            # Display the first 500 characters of the removed conversation
            preview = str(removed_conversation)[:500]
            messagebox.showinfo("Removed Conversation Preview", f"First 500 characters of removed conversation:\n\n{preview}")
        else:
            messagebox.showinfo("Info", "No conversation history to remove.")

    def update_status(self, message):
        """
        Updates the status label with the given message.

        Parameters:
        - message: str, the status message to display.
        """
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Status: {message}")
        else:
            print(f"Status update: {message}")

    def select_image(self):
        """
        Allows the user to select one or two images for the assistant's avatar.
        Updates the configuration and saves it.
        """
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])

        if not file_paths:
            return  # User cancelled the selection

        if len(file_paths) == 1:
            # Single image selected
            self.config['image_path_1'] = file_paths[0]
            self.config['image_path_2'] = file_paths[0]
            self.load_image(file_paths[0])
            messagebox.showinfo("Image Selected", "One image selected. It will be used for both default images. No animation will be visible.")
        elif len(file_paths) == 2:
            # Two images selected
            self.config['image_path_1'] = file_paths[0]
            self.config['image_path_2'] = file_paths[1]
            self.load_image(file_paths[0])  # Load the first image initially
            messagebox.showinfo("Images Selected", "Two images selected. They will be used for default image 1 and 2 respectively. An animation will be displayed when the assistant speaks.")
        else:
            # More than two images selected
            messagebox.showerror("Too Many Images", "Please select only one or two images.")
            return

        # Save the updated configuration
        self.save_config()
        self.update_config_textbox()

        # Update animation interval if it exists
        if hasattr(self, 'animation_interval_entry'):
            try:
                self.animation_interval = float(self.animation_interval_entry.get())
                self.config['animation_interval'] = self.animation_interval
            except ValueError:
                messagebox.showwarning("Invalid Interval", "Please enter a valid number for the animation interval. Using default value.")

        # Reinitialize animation settings
        self.animation_running = False
        self.current_image = 0

        if self.config['image_path_1'] != self.config['image_path_2']:
            messagebox.showinfo("Animation Enabled", "Two different images selected. Animation is now enabled.")
        else:
            messagebox.showinfo("Animation Disabled", "Same image selected for both. Animation is disabled.")

    def update_animation_interval(self, event=None):
        """
        Updates the animation interval based on the user's input.

        Parameters:
        - event: The event triggering the update (e.g., focus out event).
        """
        try:
            new_interval = float(self.animation_interval_entry.get())
            with self.animation_lock:
                self.animation_interval = new_interval
                self.config['animation_interval'] = new_interval
            self.save_config()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for the animation interval.")

    def start_animation(self):
        """
        Starts the image animation by toggling between two images.
        """
        if self.animation_running:
            return
        self.animation_running = True
        self.animation_event.clear()
        self.animation_thread = threading.Thread(target=self._animate, daemon=True)
        self.animation_thread.start()


    def monitor_animation(self):
        """
        Continuously monitors audio playback and stops animation when necessary.
        """
        while True:
            if self.animation_running and not self.audio_player.is_audio_active():
                self.after(0, self.stop_animation)  # Schedule stop_animation to run in the main thread
            time.sleep(0.1)  # Check every 100ms

    def stop_animation(self):
        """
        Stops the image animation and resets to the default image.
        """
        self.animation_running = False
        self.animation_event.set()
        if self.animation_thread:
            self.animation_thread.join()

        image_path = self.config.get('image_path_1', DEFAULT_IMAGE1_PATH)
        self.after(0, lambda: self.load_image(image_path))

    def _animate(self):
        """
        Animation loop that alternates between two images.
        """
        while self.animation_running:
            self.current_image = 1 - self.current_image  # Toggle between 0 and 1
            image_path = self.config.get(f'image_path_{self.current_image + 1}', DEFAULT_IMAGE1_PATH)
            self.after(0, lambda: self.load_image(image_path))

            # Use the event to wait, allowing for interruption
            self.animation_event.wait(timeout=self.animation_interval)

    def setup_audio(self):
        """
        Sets up the audio player and initializes audio configurations.
        """
        self.audio_player = AudioPlayer(self.on_sentence_playback_completed)  # Create an AudioPlayer instance
        self.p = pyaudio.PyAudio()  # Initialize PyAudio
        self.stream = None  # Audio input stream
        self.is_recording = False  # Flag to indicate if recording is active
        self.frames = []  # List to hold audio frames
        # Load the voice activity detection model
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.recording_lock = threading.Lock()  # Lock for thread-safe recording
        self.stop_event = threading.Event()  # Event to signal stopping of recording

    def load_config(self):
        """
        Loads the configuration from 'client_config.json' file or uses default settings.
        """
        try:
            # Try to load the config file
            with open('client_config.json', 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # If file not found, use default config
            self.config = self.get_default_config()
        self.conversation_history = self.config.get('Conversation_History', [])  # Load conversation history


    def extract_lm_activated_skill_info(self):
        # Regular expression pattern to match lines starting with "# LM ACTIVATED SKILL:"
        # It allows for any number of spaces before and after the colon

        input_string = self.concatenate_skills("./skills/")

        pattern = r'^\s*#\s*LM\s*ACTIVATED\s*SKILL\s*:.*$'
        
        # Find all matching lines in the input string
        matches = re.findall(pattern, input_string, re.MULTILINE)
        
        return '\n'.join(matches)

    def read_systemprompt(self, file_path):
        """
        Reads the system prompt from a file.

        Parameters:
        - file_path: str, the path to the system prompt file.

        Returns:
        - The content of the file as a string.
        """
        try:
            with open(file_path, 'r') as file:
                systemprompt = file.read()
                lm_activated_skill_usage_info = self.extract_lm_activated_skill_info()
                systemprompt =systemprompt.replace("###SKILL_USAGE_INSTRUCTIONS_HERE###", lm_activated_skill_usage_info)
                print("##############", systemprompt)
            return systemprompt
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_default_config(self):
        """
        Returns a default configuration dictionary.
        """
        try:
            config = {
                'LLM_Config': {
                    'model': "llama-3.1-70b-versatile",
                    'temperature': 0.6,
                    'top_p': 0.95,
                    'max_tokens': 400,
                    'frequency_penalty': 1.0,
                    'presence_penalty': 1.0
                },
                'TTS_Config': {
                    'voice': 'Stefanie',
                    'speed': 'normal'
                },
                'Skills': self.concatenate_skills("./skills/"),  # Load skills from the 'skills' directory
                'Conversation_History': [],
                'Scratchpad': {},
                'System_Prompt': self.read_systemprompt('systemprompt.txt'),  # Read the system prompt from a file
                'API_Key': '12345',
                'client_id': self.config.get('client_id', ''),
                'image_path_1': DEFAULT_IMAGE1_PATH,
                'image_path_2': DEFAULT_IMAGE2_PATH,
                'animation_interval': 0.8
            }
        except:
            # if the client_config.json got deleted, we need first to request a client_id
            self.request_client_id()
            config = {
                'LLM_Config': {
                    'model': "llama-3.1-70b-versatile",
                    'temperature': 0.6,
                    'top_p': 0.95,
                    'max_tokens': 400,
                    'frequency_penalty': 1.0,
                    'presence_penalty': 1.0
                },
                'TTS_Config': {
                    'voice': 'Stefanie',
                    'speed': 'normal'
                },
                'Skills': self.concatenate_skills("./skills/"),  # Load skills from the 'skills' directory
                'Conversation_History': [],
                'Scratchpad': {},
                'System_Prompt': self.read_systemprompt('systemprompt.txt'),  # Read the system prompt from a file
                'API_Key': '12345',
                'client_id': self.config.get('client_id', ''),
                'image_path_1': DEFAULT_IMAGE1_PATH,
                'image_path_2': DEFAULT_IMAGE2_PATH,
                'animation_interval': 0.8
            }

        return config

    def concatenate_skills(self, directory):
        """
        Reads and concatenates all skill files in the given directory.

        Parameters:
        - directory: str, the path to the skills directory.

        Returns:
        - A string containing the concatenated skill scripts.
        """
        skills = ""
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                with open(os.path.join(directory, filename), 'r') as file:
                    skills += file.read() + "\n"
        return skills

    def start_reconnection_thread(self):
        """
        Starts a background thread that attempts to reconnect to the server if disconnected.
        """
        if self.reconnection_thread is None or not self.reconnection_thread.is_alive():
            self.reconnection_thread = threading.Thread(target=self.reconnection_loop, daemon=True)
            self.reconnection_thread.start()

    def reconnection_loop(self):
        """
        Continuously checks the server connection and attempts to reconnect if disconnected.
        """
        while True:
            if not self.server_connected:
                self.connect_to_server()
            time.sleep(2)  # Wait for 2 seconds before trying to reconnect

    def connect_to_server(self):
        """
        Attempts to connect to the server and obtain a client ID.
        Updates the server connection status accordingly.
        """
        if 'API_Key' not in self.config or not self.config['API_Key']:
            messagebox.showerror("API Key Error", "Please set a valid API key in the configuration.")
            return

        try:
            response = requests.get(SERVER_IP + "/", headers={'X-API-Key': str(self.config.get('API_Key', '12345'))}, timeout=5)
            if response.status_code == 200:
                self.update_status("Connected to server")
                self.server_connected = True

                self.request_client_id() # get new client_id
                print(self.config.get('client_id'))
                if self.config.get('client_id'):
                    self.update_client_config()
                else:
                    raise Exception("Failed to obtain a valid client ID")
            else:
                raise Exception(f"Server returned status code: {response.status_code}")
        except requests.RequestException as e:
            self.update_status("Failed to connect to server")
            self.server_connected = False
            logging.error(f"Connection error: {str(e)}")

    def request_client_id(self):
        """
        Requests a client ID from the server and updates the configuration.
        """
        try:
            response = requests.post(SERVER_IP + "/generate_client_id",
                                     json=self.config,
                                     headers={'X-API-Key': str(self.config.get('API_Key', '12345'))})
            if response.status_code == 200:
                self.config['client_id'] = response.json()["client_id"]
                self.save_config()
                logging.info(f"Obtained client ID: {self.config['client_id']}")
            else:
                raise Exception(f"Failed to obtain client ID. Server returned: {response.status_code}")
        except Exception as e:
            logging.error(f"Error in request_client_id: {str(e)}")
            raise

    def update_client_config(self):
        """
        Sends the current client configuration to the server for updates.
        """
        try:
            print("self.config", self.config)
            response = requests.post(
                f"{SERVER_IP}/update_client_config/{self.config['client_id']}",
                json=self.config,
                headers={'X-API-Key': str(self.config.get('API_Key', '12345'))}
            )

            if response.status_code == 200:
                logging.info("Client configuration updated successfully on the server")
            else:
                logging.error(f"Failed to update client configuration on the server. Status: {response.status_code}")
                messagebox.showerror("Error", "Failed to update client configuration on the server. Deleting the client_config.json might help. It will be reset then to default.")
        except Exception as e:
            logging.error(f"Failed to update config: {str(e)}")
            messagebox.showerror("Error", f"Failed to update config: {str(e)}")

    def toggle_recording(self):
        """
        Toggles the recording state between start and stop.
        """
        if not self.inConversation:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """
        Starts recording audio input and processing it.
        """
        with self.recording_lock:
            if self.inConversation:
                return
            try:
                self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=512)
                self.inConversation = True
                self.stop_event.clear()
                self.conversation_button.config(text="Stop Conversation", bg="#FFA500")  # Orange color when active
                if hasattr(self, 'start_button'):
                    self.start_button.config(text="Stop Conversation", bg="#FFA500")  # Orange color when active
                self.status_label.config(text="Status: Recording...")
                threading.Thread(target=self.process_audio, daemon=True).start()
            except Exception as e:
                logging.error(f"Failed to start recording: {str(e)}")
                messagebox.showerror("Audio Error", "Failed to start recording. Please check your audio device.")

    def stop_recording(self):
        """
        Stops recording audio input.
        """
        with self.recording_lock:
            if not self.inConversation:
                return
            self.inConversation = False
            self.stop_playback()
            self.stop_event.set()
            if self.stream:
                self.stream.stop_stream()
            self.conversation_button.config(text="Start Conversation", bg="SystemButtonFace")  # Default color when inactive
            if hasattr(self, 'start_button'):
                self.start_button.config(text="Start Conversation", bg="SystemButtonFace")  # Default color when inactive
            self.status_label.config(text="Status: Stopped")

    def stop_playback(self):
        """
        Stops audio playback.
        """
        logging.info("Stopping playback")
        self.audio_player.stop_playback()

        self.status_label.config(text="Status: Playback stopped")
        if self.inConversation:
            self.stop_recording()

    def on_closing(self):
        """
        Handles cleanup when the application window is closed.
        """
        self.server_connected = False  # Stop the reconnection loop
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
        """
        Processes audio input, detects voice activity, and sends audio segments to the server.
        """
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
                            if silent_count >= RECORDING_TIME_AFTER_LAST_VOICE_ACTIVITY_DETECTED:
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

    def process_sentence(self, sentence):
        """
        Converts a text sentence to speech and queues it for playback.

        Parameters:
        - sentence: str, the text to convert to speech.
        """
        if not sentence or not isinstance(sentence, str):
            logging.warning(f"Invalid sentence received: {sentence}")
            return

        try:
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
            audio_content = response.content

            sentence_audio = SentenceAudio(
                text=sentence,
                audio_data=BytesIO(audio_content),
                sample_rate=16000,
                channels=1
            )

            self.start_animation()
            self.audio_player.queue_audio(sentence_audio)

        except Exception as e:
            logging.error(f"Error in process_sentence: {str(e)}", exc_info=True)
            messagebox.showerror("TTS Error", "An error occurred while generating speech. Please try again.")

    def send_audio_segment_to_server(self, audio_data):
        """
        Sends an audio segment to the server for processing and handles the response.

        Parameters:
        - audio_data: bytes, the raw audio data to send.
        """
        try:
            clipboard_content = get_clipboard_content()
            if isinstance(clipboard_content, Image.Image):
                clipboard_data = process_image(clipboard_content)
            elif clipboard_content:
                clipboard_data = {
                    "type": "text",
                    "data": clipboard_content
                }
            else:
                clipboard_data = None
            self.config["clipboard"] = json.dumps(clipboard_data)

            self.update_client_config()  # Ensure the server has the latest config before sending audio

            with BytesIO() as wav_file:
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

                response = requests.post(f"{SERVER_IP}/receive_audio",
                                         files=files,
                                         data=form_data,
                                         headers=headers)

                response.raise_for_status()
                response_content = response.content

            parts = response_content.split(b'\n---AUDIO_DATA---\n')
            if len(parts) == 2:
                json_data, audio_data = parts
                data = json.loads(json_data.decode('utf-8'))
                new_conversation_history = data.get('updated_conversation_history', self.conversation_history)
                self.update_conversation_history(new_conversation_history)

             
                config_updates = data.get('config_updates', {})
                if config_updates:
                    self.config.update(config_updates)
                    self.save_config()
                    self.update_config_textbox()

                sentences = data.get('sentences') or [data.get('response')]

                if self.config.get("code_for_client_execution") and len(self.config["code_for_client_execution"]) > 0:
                    execute_client_code(self.config["code_for_client_execution"])
                    self.config["code_for_client_execution"] = ""
                self.start_animation()
                if sentences and isinstance(sentences[0], str):
                    sentence_audio = SentenceAudio(
                        text=sentences[0],
                        audio_data=BytesIO(audio_data),
                        sample_rate=16000,
                        channels=1
                    )
                    self.start_animation()
                    self.audio_player.queue_audio(sentence_audio)

                    for sentence in sentences[1:]:
                        if isinstance(sentence, str):
                            s1=time.time()
                            self.process_sentence(sentence)
                            print("TTS ENDPOINT LATENCY:", time.time()-s1)
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
            if hasattr(e, 'response') and e.response:
                logging.error(f"Response status code: {e.response.status_code}")
                logging.error(f"Response headers: {e.response.headers}")
                logging.error(f"Response content: {e.response.text}")
            messagebox.showerror("Server Error", f"Failed to communicate with the server: {str(e)}")
        except Exception as e:
            logging.error(f"Error in send_audio_segment_to_server: {str(e)}", exc_info=True)
            messagebox.showerror("Error", "An error occurred while processing your request.")

    def on_sentence_playback_completed(self, sentence_text):
        """
        Callback function called when a sentence playback is completed.

        Parameters:
        - sentence_text: str, the text of the sentence that was played.
        """
        logging.info(f"Completed playback of sentence: {sentence_text[:30]}...")
        if self.audio_player.audio_queue.empty():
            self.stop_animation()

    def load_config_file(self):
        """
        Loads a configuration file selected by the user.
        """
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
        """
        Saves the current configuration to a file selected by the user.
        """
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
        """
        Clears the current configuration and resets to default.
        """
        self.config = self.get_default_config()
        self.conversation_history = []
        self.config['Conversation_History'] = []
        self.update_client_config()
        self.update_config_textbox()
        self.save_config()
        messagebox.showinfo("Config Cleared", "Configuration has been reset to default.")

    def update_config_textbox(self):
        """
        Updates the configuration text box with the current configuration.
        """
        if hasattr(self, 'config_textbox'):
            self.config_textbox.delete(1.0, tk.END)
            config_to_display = self.config.copy()
            config_to_display['Conversation_History'] = self.conversation_history
            try:
                config_str = json.dumps(config_to_display, indent=4, default=self.json_default)
                self.config_textbox.insert(tk.END, config_str)
            except Exception as e:
                error_msg = f"Error serializing config: {str(e)}"
                self.config_textbox.insert(tk.END, error_msg)
                logging.error(error_msg)

    def json_default(self, obj):
        """
        Default JSON serialization method for unsupported types.

        Parameters:
        - obj: The object to serialize.
        """
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    def on_config_change(self, event):
        """
        Event handler for changes in the configuration text box.
        """
        try:
            config_text = self.config_textbox.get(1.0, tk.END)
            new_config = json.loads(config_text)
            self.config = new_config
            self.update_conversation_history(new_config.get('Conversation_History', []))
            self.config_status_label.config(text="Config Status: OK", fg="green")
        except json.JSONDecodeError:
            self.config_status_label.config(text="Config Status: Invalid JSON", fg="red")
        except Exception as e:
            self.config_status_label.config(text=f"Config Status: Error - {str(e)}", fg="red")


    def save_config(self):
        """
        Saves the current configuration to 'client_config.json'.
        """
        # Update animation interval from entry if it exists
        if hasattr(self, 'animation_interval_entry'):
            try:
                self.config['animation_interval'] = float(self.animation_interval_entry.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for the animation interval.")

        with open('client_config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def setup_wake_word(self):
        """
        Sets up the wake word detection using Porcupine.

        Note: This function assumes the Porcupine keyword files are available.
        """
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
        """
        Continuous loop that listens for wake words and triggers actions.
        """
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
                        self.last_wake_word_time = current_time
                        self.after(0, self.activate_conversation_mode)
                        time.sleep(0.5)
                elif keyword_index == 1:  # "Stop Buddy" detected
                    self.after(0, self.deactivate_conversation_mode)
        except Exception as e:
            print(f"Error in wake word detection loop: {e}")
        finally:
            audio_stream.close()
            pa.terminate()

    def activate_conversation_mode(self):
        """
        Activates conversation mode via wake word.
        """
        if not self.inConversation:
            self.toggle_recording()

    def deactivate_conversation_mode(self):
        """
        Deactivates conversation mode via stop word.
        """
        if self.inConversation:
            self.toggle_recording()
        self.stop_playback()

if __name__ == "__main__":
    # Create an instance of the BudEClient
    app = BudEClient()
    # Set the protocol for closing the application
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    # Start the main event loop
    app.mainloop()
