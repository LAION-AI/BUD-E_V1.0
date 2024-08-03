import asyncio
import pyaudio
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, Response
import uvicorn
import threading
import io
import webbrowser
import pyautogui
import os
import mimetypes
import requests
import json
import hashlib
import pandas as pd
from pydub import AudioSegment
import time
import nltk
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel

nltk.download('punkt', quiet=True)

from bud_e_transcribe import transcribe
from bud_e_tts import text_to_speech
from bud_e_llm import ask_LLM

app = FastAPI()

client_sessions = {}
is_running = True

def generate_client_id():
    return hashlib.sha256(os.urandom(10)).hexdigest()[:10]

def split_text_into_sentences(text):
    sentences = sent_tokenize(text)
    merged_sentences = []
    current_sentence = ""
    
    for i, sentence in enumerate(sentences):
        if not merged_sentences:
            # For the first merged sentence
            current_sentence += sentence + " "
            if len(current_sentence) >= 40 or i == len(sentences) - 1:
                merged_sentences.append(current_sentence.strip())
                current_sentence = ""
        else:
            # For subsequent sentences
            if len(current_sentence) + len(sentence) <= 300:
                current_sentence += sentence + " "
            else:
                if len(current_sentence) >= 200:
                    merged_sentences.append(current_sentence.strip())
                    current_sentence = sentence + " "
                else:
                    current_sentence += sentence + " "
            
            # Check if it's the last sentence
            if i == len(sentences) - 1:
                if len(current_sentence) >= 200:
                    merged_sentences.append(current_sentence.strip())
                elif merged_sentences:
                    merged_sentences[-1] += " " + current_sentence.strip()
    
    # If there's only one short sentence in the entire text
    if not merged_sentences and current_sentence:
        merged_sentences.append(current_sentence.strip())
    
    return merged_sentences
@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.post("/generate_client_id")
async def generate_client_id_endpoint():
    new_client_id = generate_client_id()
    client_sessions[new_client_id] = {
        'LLM-Config': {
            'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
            'temperature': 0.7,
            'top_p': 0.95,
            'max_tokens': 400,
            'frequency_penalty': 1.1,
            'presence_penalty': 1.1
        },
        'TTS-Config': {'voice': 'en-us'},
        'Skills': ['edit', 'completion'],
        'Conversation History': [],
        'Scratchpad': {},
        'System Prompt': 'Initial Prompt'
    }
    return JSONResponse(content={"client_id": new_client_id}, status_code=200)
@app.post("/receive_audio")
async def receive_audio(client_id: str = Query(...), file: UploadFile = File(...)):
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    filename = f"{client_id}_audio.wav"
    with open(filename, "wb") as audio_file:
        audio_file.write(await file.read())

    try:
        start_time = time.time()
        user_input = transcribe(filename, client_id)
        end_time = time.time()
        print(f"ASR Time: {end_time - start_time} seconds")
        print(f"User: {user_input}")
    finally:
        os.remove(filename)  # Delete the audio file after transcription

    client_session = client_sessions[client_id]
    conversation_history = client_session.get('Conversation History', [])
    system_prompt = client_session.get('System Prompt', '')
    llm_config = client_session.get('LLM-Config', {})

    conversation_history.append({"role": "user", "content": user_input})

    start_time = time.time()
    ai_response = ask_LLM(
        llm_config['model'],
        system_prompt,
        str(conversation_history)+" - Write BUD-E's reply to the user without chat formarting in brackets and no role, just directly the reply:",
        temperature=llm_config['temperature'],
        top_p=llm_config['top_p'],
        max_tokens=llm_config['max_tokens'],
        frequency_penalty=llm_config['frequency_penalty'],
        presence_penalty=llm_config['presence_penalty']
    )
    end_time = time.time()
    print(f"LLM Time: {end_time - start_time} seconds")
    print(f"AI: {ai_response}")

    conversation_history.append({"role": "assistant", "content": ai_response})

    # Update the conversation history in the client session
    client_sessions[client_id]['Conversation History'] = conversation_history

    sentences = split_text_into_sentences(ai_response)
    
    first_sentence_audio = None
    if sentences:
        first_sentence_audio = await generate_tts(sentences[0])
    
    response_data = {
        "first_sentence_audio": first_sentence_audio,
        "sentences": sentences,
        "updated_conversation_history": conversation_history,
        "config_updates": client_session  # Send the full session data as config updates
    }

    return JSONResponse(content=response_data, status_code=200)


async def generate_tts(sentence: str):
    tts_output = text_to_speech(sentence)
    if tts_output is None:
        print(f"TTS generation failed for sentence: {sentence}")
        return None
    
    tts_filename = f"tts_output_{hash(sentence)}.wav"
    with open(tts_filename, 'wb') as tts_file:
        tts_file.write(tts_output)
    return tts_filename

class SentenceRequest(BaseModel):
    sentence: str

@app.post("/generate_tts")
async def generate_tts_endpoint(request: SentenceRequest):
    tts_filename = await generate_tts(request.sentence)
    if tts_filename is None:
        raise HTTPException(status_code=500, detail="TTS generation failed")
    return {"filename": tts_filename}

class DeleteFileRequest(BaseModel):
    filename: str

@app.post("/delete_tts_file")
async def delete_tts_file(request: DeleteFileRequest):
    try:
        if os.path.exists(request.filename):
            os.remove(request.filename)
            return JSONResponse(content={"message": "File deleted successfully"}, status_code=200)
        else:
            return JSONResponse(content={"message": "File not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"message": f"Error deleting file: {str(e)}"}, status_code=500)

@app.get('/take_screenshot')
async def take_screenshot(client_id: str):
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    start_time = time.time()
    screenshot = pyautogui.screenshot()
    img_byte_arr = io.BytesIO()
    screenshot.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    end_time = time.time()
    print(f"Screenshot Time: {end_time - start_time} seconds")
    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png", headers={"Content-Disposition": "attachment; filename=screenshot.png"})

@app.post('/open_website')
async def open_website(client_id: str, url: str):
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    start_time = time.time()
    webbrowser.open(url)
    end_time = time.time()
    print(f"Open Website Time: {end_time - start_time} seconds")
    return JSONResponse(content={"message": f"Opened website: {url}"}, status_code=200)

@app.get("/send_file")
async def send_file(client_id: str, file: str):
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    if not os.path.exists(file):
        raise HTTPException(status_code=404, detail="File not found")

    start_time = time.time()
    with open(file, 'rb') as f:
        file_content = f.read()
    end_time = time.time()
    print(f"Read File Time: {end_time - start_time} seconds")

    mime_type, _ = mimetypes.guess_type(file)
    return Response(content=file_content, media_type=mime_type, headers={
        "Content-Disposition": f'attachment; filename="{os.path.basename(file)}"'
    })

@app.post("/update_client_config/{client_id}")
async def update_client_config(client_id: str, config: dict):
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    client_sessions[client_id].update(config)

    return JSONResponse(content={"message": "Client configuration updated successfully"}, status_code=200)

@app.get("/get_client_data/{client_id}")
async def get_client_data_endpoint(client_id: str):
    if client_id not in client_sessions:
        raise HTTPException(status_code=404, detail="Client not found")
    return JSONResponse(content=client_sessions[client_id], status_code=200)

def save_and_play_audio(client_id):
    """Function to save and play the latest audio file for a client."""
    client_session = client_sessions.get(client_id)
    if not client_session:
        print("Invalid client ID")
        return

    audio_files = client_session.get('audio_files', [])
    if not audio_files:
        print("No audio data to save and play.")
        return

    print("Playing the latest audio file...")
    latest_audio_file = audio_files[-1]

    p = pyaudio.PyAudio()
    wf = wave.open(latest_audio_file, 'rb')
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

    print("Audio buffer cleared")

def print_menu():
    """Function to print the server menu."""
    print("\n1. Save and Play Audio")
    print("2. Take Screenshot")
    print("3. Open Website")
    print("4. Send File")
    print("5. Exit")
    print("Enter your choice (1-5): ", end="", flush=True)

def run_cli():
    """Function to run the command-line interface."""
    global is_running
    while is_running:
        print_menu()
        choice = input()

        if choice == "1":
            client_id = input("Enter client ID: ")
            save_and_play_audio(client_id)
        elif choice == "2":
            client_id = input("Enter client ID: ")
            screenshot = pyautogui.screenshot()
            screenshot.save("server_screenshot.png")
            print("Screenshot saved as server_screenshot.png")
        elif choice == "3":
            client_id = input("Enter client ID: ")
            url = input("Enter the URL to open: ")
            webbrowser.open(url)
            print(f"Opened website: {url}")
        elif choice == "4":
            client_id = input("Enter client ID: ")
            file_path = input("Enter the path of the file to send: ")
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    files = {"file": (os.path.basename(file_path), file)}
                    response = requests.get(f"http://0.0.0.0:8006/send_file?client_id={client_id}&file={file_path}")
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

def run_server():
    """Function to run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    # Start the FastAPI server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Run the CLI in the main thread
    run_cli()

    print("Shutting down...")
