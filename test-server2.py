# Import necessary libraries
import asyncio  # For asynchronous programming
import pyaudio  # For audio processing
import wave  # For working with WAV files
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile  # FastAPI framework and WebSocket
from fastapi.responses import JSONResponse, StreamingResponse, Response  # Different types of responses
import uvicorn  # ASGI server
from typing import List  # For type hinting
import threading  # For running multiple threads
import io  # For working with byte streams
import webbrowser  # For opening web browsers
import pyautogui  # For taking screenshots
import os  # For file and path operations
import mimetypes  # For determining MIME types of files
import requests  # For making HTTP requests
import torch
import numpy as np
from pydub import AudioSegment

import requests
import sys
import os

# Initialize FastAPI application
app = FastAPI()

# Global variables
RawAudioBuffer: List[bytes] = []  # Buffer to store raw audio data
is_running = True  # Flag to control the main loop
first_chunk_received = False  # Flag to track if the first audio chunk has been received

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Server is running"}  # Return a simple JSON response

# WebSocket endpoint for audio streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global first_chunk_received
    await websocket.accept()  # Accept the WebSocket connection
    print("WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_bytes()  # Receive audio data as bytes
            RawAudioBuffer.append(data)  # Add the received data to the buffer
            if not first_chunk_received:
                print(f"Received audio chunk of size: {len(data)} bytes")
                first_chunk_received = True

            # Perform VAD on the received audio chunk
            await vad_on_audio(data)
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket error: {e}")

async def vad_on_audio(data):
    files = {'data': ('audio_chunk.wav', io.BytesIO(data), 'audio/wav')}
    response = requests.post("http://localhost:8001/vad", files=files)
    if response.status_code == 200:
        confidence = response.json()["confidence"]
        print(confidence)
        if confidence > 0.4:
            print("*", end="", flush=True)  # Print * if voice activity is detected with high confidence
        else:
            print("-", end="", flush=True)  # Print - if voice activity is not detected or has low confidence
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Endpoint to get the recorded audio
@app.get('/get_audio')
async def get_audio():
    if not RawAudioBuffer:
        return JSONResponse(content={"error": "No audio data available"}, status_code=404)
    
    def audio_stream():
        with io.BytesIO() as buffer:
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)  # Mono audio
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(44100)  # 44.1kHz sample rate
                wf.writeframes(b''.join(RawAudioBuffer))
            buffer.seek(0)
            yield from buffer

    return StreamingResponse(audio_stream(), media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=recorded_audio.wav"})

# Endpoint to take a screenshot
@app.get('/take_screenshot')
async def take_screenshot():
    screenshot = pyautogui.screenshot()  # Take a screenshot
    img_byte_arr = io.BytesIO()  # Create a byte stream
    screenshot.save(img_byte_arr, format='PNG')  # Save the screenshot as PNG
    img_byte_arr = img_byte_arr.getvalue()  # Get the byte value
    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png", headers={"Content-Disposition": "attachment; filename=screenshot.png"})

# Endpoint to open a website
@app.post('/open_website')
async def open_website(url: str):
    webbrowser.open(url)  # Open the specified URL in the default web browser
    return JSONResponse(content={"message": f"Opened website: {url}"}, status_code=200)

# New endpoint to send a file to the client
@app.post("/send_file")
async def send_file(file: UploadFile = File(...)):
    file_content = await file.read()  # Read the file content
    
    # Determine the MIME type of the file
    mime_type, _ = mimetypes.guess_type(file.filename)
    
    # Create a Response with the file content
    response = Response(content=file_content, media_type=mime_type)
    
    # Set the Content-Disposition header
    response.headers["Content-Disposition"] = f'attachment; filename="{file.filename}"'
    
    return response

# Function to save and play the recorded audio
def save_and_play_audio():
    global RawAudioBuffer
    if not RawAudioBuffer:
        print("No audio data to save and play.")
        return

    print(f"Total audio chunks: {len(RawAudioBuffer)}")
    print(f"Total audio size: {sum(len(chunk) for chunk in RawAudioBuffer)} bytes")

    with wave.open("streamed-audio.wav", "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(44100)  # 44.1kHz sample rate
        wf.writeframes(b''.join(RawAudioBuffer))

    print("Audio saved as streamed-audio.wav")

    # Play the audio
    p = pyaudio.PyAudio()
    wf = wave.open("streamed-audio.wav", 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Clear the buffer after saving and playing
    RawAudioBuffer.clear()
    print("Audio buffer cleared")

# Function to run the FastAPI server
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Function to print the CLI menu
def print_menu():
    print("\n1. Save and Play Audio")
    print("2. Take Screenshot")
    print("3. Open Website")
    print("4. Send File")
    print("5. Exit")
    print("Enter your choice (1-5): ", end="", flush=True)

# Function to run the CLI interface
def run_cli():
    global is_running, first_chunk_received
    while is_running:
        print_menu()
        choice = input()

        if choice == "1":
            save_and_play_audio()
        elif choice == "2":
            screenshot = pyautogui.screenshot()
            screenshot.save("server_screenshot.png")
            print("Screenshot saved as server_screenshot.png")
        elif choice == "3":
            url = input("Enter the URL to open: ")
            webbrowser.open(url)
            print(f"Opened website: {url}")
        elif choice == "4":
            file_path = input("Enter the path of the file to send: ")
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    files = {"file": (os.path.basename(file_path), file)}
                    response = requests.post("http://localhost:8000/send_file", files=files)
                    if response.status_code == 200:
                        print(f"File sent successfully: {file_path}")
                    else:
                        print(f"Failed to send file: {response.text}")
            else:
                print("File not found.")
        elif choice == "5":
            is_running = False
            break
        else:
            print("Invalid choice. Please try again.")
        
        first_chunk_received = False  # Reset for next audio stream

# Main entry point of the script
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    cli_thread = threading.Thread(target=run_cli)
    cli_thread.start()

    cli_thread.join()
    print("Shutting down...")
