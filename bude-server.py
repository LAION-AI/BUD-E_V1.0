
'''

import subprocess
import sys

def install_dependencies():
    dependencies = [
        'asyncio',
        'pyaudio',
        'fastapi',
        'uvicorn',
        'pyautogui',
        'requests',
        'pandas',
        'pydub',
        'nltk',
        'pydantic',
        'websockets',  # For WebSocket support in FastAPI
    ]

    print("Installing dependencies...")

    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"{dep} installed successfully.")<
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}. Please install it manually.")

    # Additional setup for NLTK
    print("Downloading NLTK punkt tokenizer...")
    import nltk
    nltk.download('punkt')

    print("\nAll dependencies have been installed.")
    print("Note: Some packages like 'pyaudio' might require additional system-level dependencies.")
    print("If you encounter any issues, please refer to the documentation of the respective packages.")

if __name__ == "__main__":
    install_dependencies()
'''
import asyncio
import pyaudio
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse

import uvicorn
import threading
import io
import webbrowser
# import pyautogui
import os
import mimetypes
import requests
import json
import hashlib
import pandas as pd
from pydub import AudioSegment
import sys
import time
import nltk
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from langdetect import detect, LangDetectException

nltk.download('punkt', quiet=True)

from bud_e_transcribe import transcribe
from bud_e_tts import text_to_speech
from bud_e_llm import ask_LLM
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

import sys
from types import ModuleType
import json
import re


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


logger = logging.getLogger(__name__)



from langdetect import detect, LangDetectException

def detect_language(text):
    try:
        # Detect the language
        lang = detect(text)
        return lang
    except LangDetectException:
        return "Unknown"
    except Exception as e:
        print(f"An error occurred during language detection: {e}")
        return "Error"


class SentenceRequest(BaseModel):
    sentence: str
    client_id: str
    voice: str
    speed: str

class ClientConfig(BaseModel):
    LLM_Config: dict
    TTS_Config: dict
    Skills: str
    Conversation_History: list
    Scratchpad: dict
    System_Prompt: str


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
            if len(current_sentence) >= 20 or i == len(sentences) - 1:
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


def create_module_from_code(code, module_name="dynamic_skills"):
    module = ModuleType(module_name)
    exec(code, module.__dict__)
    return module



def parse_lm_activated_skills(code_string):
    # Regular expression to match LM ACTIVATED SKILL comments and function definitions
    pattern = r'# LM ACTIVATED SKILL:.*?USAGE INSTRUCTIONS:.*?<(\w+)>(.*?)</\1>.*?\n(def\s+(\w+)\(.*?\):.*?)\n\n'
    
    # Find all matches in the code string
    matches = re.findall(pattern, code_string, re.DOTALL)
    
    # Dictionary to store the results
    skills = {}
    
    # Process each match
    for match in matches:
        tag_name, _, function_def, function_name = match
        opening_tag = f"<{tag_name}>"
        closing_tag = f"</{tag_name}>"
        
        # Add to the skills dictionary
        skills[function_name] = {
            "opening_tag": opening_tag,
            "closing_tag": closing_tag,
        'function_name': function_name

        }
    
    return skills


def extract_skill_calls(ai_response, lm_activated_skills):
    skill_calls = []
    for skill_title, skill_info in lm_activated_skills.items():
        opening_tag = re.escape(skill_info['opening_tag'])
        closing_tag = re.escape(skill_info['closing_tag'])
        pattern = rf'{opening_tag}(.*?){closing_tag}'
        matches = re.findall(pattern, ai_response, re.DOTALL)
        for content in matches:
            skill_calls.append({
                "name": skill_title,
                "function_name": skill_info['function_name'],
                "parameters": content.strip()
            })
    return skill_calls
import importlib
import sys
from types import ModuleType
import traceback

def import_skills_code(skills_code):
    # Create a new module to hold our skills
    skills_module = ModuleType("dynamic_skills")
    
    # Add the module to sys.modules so it can be imported
    sys.modules["dynamic_skills"] = skills_module
    
    # Dictionary to hold successfully imported functions
    imported_functions = {}
    
    try:
        # Execute the skills code in the context of our new module
        exec(skills_code, skills_module.__dict__)
        
        # Iterate through the module's attributes
        for attr_name in dir(skills_module):
            attr = getattr(skills_module, attr_name)
            # Check if it's a function and doesn't start with underscore
            if callable(attr) and not attr_name.startswith("_"):
                imported_functions[attr_name] = attr
                
        print(f"Successfully imported functions: {', '.join(imported_functions.keys())}")
    except Exception as e:
        print(f"Error importing skills code: {str(e)}")
        print(traceback.format_exc())
    
    return imported_functions


def process_lm_activated_skills(ai_response, client_session, user_input):
    skills_code = client_session.get('Skills', '')
    print(skills_code)
    lm_activated_skills = parse_lm_activated_skills(skills_code)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("lm_activated_skills:", lm_activated_skills)
    
    # Import the skills code
    imported_functions = import_skills_code(skills_code)
    print("Successfully imported functions:", ", ".join(imported_functions.keys()))
    
    skill_calls = extract_skill_calls(ai_response, lm_activated_skills)
    print("skill_calls:", skill_calls)
    
    updated_config = client_session.copy()
    
    for skill_call in skill_calls:
        skill_name = skill_call["name"]
        function_name = skill_call["function_name"]
        params = skill_call["parameters"]
        
        skill_info = lm_activated_skills.get(skill_name)
        if skill_info:
            opening_tag = re.escape(skill_info['opening_tag'])
            closing_tag = re.escape(skill_info['closing_tag'])
            
            # Check if the skill call is properly enclosed in tags in the AI response
            skill_pattern = rf'{opening_tag}(.*?){closing_tag}'
            if re.search(skill_pattern, ai_response):
                if function_name in imported_functions:
                    skill_function = imported_functions[function_name]
                    try:
                        # Call the function with the correct parameters
                        # Ensure params is in the correct format: "(param_value)"
                        formatted_params = f"({params})" if not params.startswith("(") else params
                        skill_response, updated_config = skill_function(user_input, updated_config, formatted_params)
                        
                        # Replace the skill call in the AI response with the skill response
                        ai_response = re.sub(
                            rf'{opening_tag}.*?{closing_tag}',
                            skill_response,
                            ai_response,
                            count=1
                        )
                        print(f"{skill_name} executed. Response: {skill_response}")
                    except Exception as e:
                        print(f"Error executing function {function_name}: {str(e)}")
                        print(traceback.format_exc())
                else:
                    print(f"Warning: Function {function_name} not found in imported functions.")
            else:
                print(f"Warning: Skill call for {skill_name} found, but not properly enclosed in tags in AI response.")
        else:
            print(f"Warning: Skill info for {skill_name} not found in lm_activated_skills.")
    
    # Print the updated TTS config for debugging
    print("Updated TTS CONFIG:", updated_config.get('TTS_Config', {}))
    
    return ai_response, updated_config

def skill_execution(function_name, transcription_response, client_session, LMGeneratedParameters=""):
    print(f"Executing skill: {function_name}")
    
    skills_code = client_session.get('Skills', '')
    
    # Create a module from the skills code
    module = {}
    exec(skills_code, module)
    
    # Try to get the function from the dynamically created module
    function_to_run = module.get(function_name)
    
    if function_to_run is None:
        raise ValueError(f"The specified function '{function_name}' is not defined.")
    
    if not callable(function_to_run):
        raise ValueError(f"The function '{function_name}' is not callable.")
    
    # Execute the function
    skill_response, updated_client_session = function_to_run(
        transcription_response, client_session, LMGeneratedParameters
    )
    
    print(f"Skill execution complete: {function_name}")
    return skill_response, updated_client_session


# Define a function to parse a string representation of a list of lists
def parse_list_of_lists(input_str):
    """
    Parses a string representing a list of lists, where each sublist contains strings.
    The function handles irregular spacing and variations in quote usage.
    
    Args:
    input_str (str): A string representation of a list of lists.
    
    Returns:
    list of list of str: The parsed list of lists.
    """
    # Normalize the string by replacing single quotes with double quotes
    normalized_str = re.sub(r"\'", "\"", input_str)

    # Extract the sublists using a regular expression that captures contents inside brackets
    sublist_matches = re.findall(r'\[(.*?)\]', normalized_str)
    
    # Process each match to extract individual string elements
    result = []
    for sublist in sublist_matches:
        # Extract string elements inside the quotes
        strings = re.findall(r'\"(.*?)\"', sublist)
        result.append(strings)

    return result


def extract_activated_skills_from_code(code_strings: list, keyword: str = "KEYWORD ACTIVATED SKILL:") -> dict:
    # Initialize an empty dictionary to store activated skills
    activated_skills = {}

    # Iterate through provided Python code strings
    for code_string in code_strings:
        # Split the code into lines
        lines = code_string.split('\n')

        # Search for functions with the specified keyword comment
        for i in range(len(lines) - 1):
            # Check if the current line contains the keyword (case-insensitive)
            if keyword.lower() in lines[i].lower():
                # Check if the next line is a function definition
                if re.match(r'^\s*def\s+\w+\s*\(', lines[i + 1]):
                    # Extract the function name using regex
                    function_name = re.findall(r'def\s+(\w+)\s*\(', lines[i + 1])[0]
                    # Extract the comment text after the keyword
                    comment = lines[i].strip().split(keyword)[-1].strip()

                    # Store the function name and comment in the dictionary
                    activated_skills[function_name] = comment

    # Return the dictionary of activated skills
    return activated_skills






@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.post("/generate_client_id")
async def generate_client_id_endpoint(config: ClientConfig):
    new_client_id = generate_client_id()
    client_sessions[new_client_id] = {
        'LLM_Config': config.LLM_Config,
        'TTS_Config': config.TTS_Config,
        'Skills': config.Skills,
        'Conversation_History': config.Conversation_History,
        'Scratchpad': config.Scratchpad,
        'System_Prompt': config.System_Prompt
    }
    return JSONResponse(content={"client_id": new_client_id}, status_code=200)

@app.post("/receive_audio")
async def receive_audio(
    client_id: str = Form(...),
    file: UploadFile = File(...),
    client_config: str = Form(...)
):
    startendpoint= time.time()
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    # Parse the client_config JSON string
    try:
        config_dict = json.loads(client_config)
        config = ClientConfig(**config_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in client_config")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid data in client_config")

    skills_code = config_dict.get('Skills', '')
    # Parse language model activated skills
    lm_activated_skills = parse_lm_activated_skills(skills_code)
    logging.info(f"Parsed LM activated skills: {lm_activated_skills}")

    #print("client_sessions[client_id]:", type( client_sessions[client_id]), client_sessions[client_id])
    #print("LLM_Config:", client_sessions[client_id]["LLM_Config"])
    #print("LLM_Config [model]:", client_sessions[client_id]["LLM_Config"]["model"])


    print("&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&")
    print("&&&&&&&&&&&&&&&&&&&")

    #print("client_session:", client_session)

    # Process the audio file
    filename = f"{client_id}_audio.wav"
    with open(filename, "wb") as audio_file:
        audio_file.write(await file.read())

    try:
        start_time = time.time()
        user_input = transcribe(filename, client_id)
        end_time = time.time()
        print(f"ASR Time: {end_time - start_time} seconds")
        print(f"User: {user_input}")

        start_time = time.time()
        language = detect_language(user_input)
        end_time = time.time()
        print(f"Language Detected: {language}")
        print(f"Language Detection Time: {end_time - start_time} seconds")

        
  
        keyword_activated_skills = extract_activated_skills_from_code([skills_code])

        server_side_skills = []
        client_side_skills = []

        skill_response = ""
        #print("SKILL-CODE:", skills_code)
        for skill_name, skill_comment in keyword_activated_skills.items():
            conditions_list = parse_list_of_lists(skill_comment)

            if any(all(cond.lower() in user_input.lower() for cond in condition) for condition in conditions_list):
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                if 'client_side_execution' in skill_name.lower():
                    client_side_skills.append(skill_name)
                elif 'server_side_execution' in skill_name.lower():
                    server_side_skills.append(skill_name)

        for skill_name in server_side_skills:
            skill_response, config_dict = skill_execution(
                skill_name, user_input, client_sessions[client_id])

        if client_side_skills:
            skill_response, client_session = await send_skills_to_client(client_id, client_side_skills, client_sessions[client_id])

        system_prompt = client_sessions[client_id].get('System_Prompt', '')
        llm_config = client_sessions[client_id].get('LLM_Config', {})
        print("LLM CONFIG:",llm_config )

        conversation_history = client_sessions[client_id].get('Conversation_History', [])
        if skill_response != "":
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "BUD-E (generated by skill)", "content": skill_response})
        else:
            conversation_history.append({"role": "user", "content": user_input})

        conversation_history_str = ""
        for message in conversation_history:
            if message["role"] == "user":
                conversation_history_str += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                conversation_history_str += f"BUD-E: {message['content']}\n"

        start_time = time.time()
        ai_response = ask_LLM(
            llm_config['model'],
            system_prompt,
            conversation_history_str,
            temperature=llm_config['temperature'],
            top_p=llm_config['top_p'],
            max_tokens=llm_config['max_tokens'],
            frequency_penalty=llm_config['frequency_penalty'],
            presence_penalty=llm_config['presence_penalty']
        )
        end_time = time.time()
        print(f"LLM Time: {end_time - start_time} seconds")
        print(f"AI: {ai_response}")

        # Process language model activated skills
        ai_response, updated_config_dict = process_lm_activated_skills(ai_response, client_sessions[client_id] , user_input)

        conversation_history.append({"role": "assistant", "content": ai_response})

        client_sessions[client_id] = updated_config_dict

        sentences = split_text_into_sentences(ai_response)

        tts_config = client_sessions[client_id].get('TTS_Config', {})

        voice = tts_config.get('voice', 'Stefanie')
        speed = tts_config.get('speed', 'normal')

        print("TTS CONFIG:",tts_config ) 
        print(voice, speed) 
        first_sentence_audio = None
        if sentences:
            first_sentence_audio = await generate_tts(sentences[0], voice, speed,  client_sessions[client_id])


            # Analyze first_sentence_audio
            print("Type of first_sentence_audio:", type(first_sentence_audio))
    
            if isinstance(first_sentence_audio, str):
                print("first_sentence_audio is a string. Content:", first_sentence_audio[:100])  # Print first 100 chars
            elif isinstance(first_sentence_audio, bytes):
                print("first_sentence_audio is bytes. Length:", len(first_sentence_audio))
                # Try to detect audio format
                if first_sentence_audio.startswith(b'RIFF'):
                  print("Appears to be a WAV file")
                elif first_sentence_audio.startswith(b'\xFF\xFB') or first_sentence_audio.startswith(b'ID3'):
                  print("Appears to be an MP3 file")
                else:
                  print("Unknown audio format")
            elif first_sentence_audio is None:
                print("first_sentence_audio is None")
            else:
               print("first_sentence_audio is an unexpected type")

        response_data = {
            "first_sentence_audio": first_sentence_audio,
            "sentences": sentences,
            "updated_conversation_history": conversation_history,
            "config_updates":  client_sessions[client_id]
        }
        print("TIME FULL ENDPOINT SCRIPT:" , time.time()-startendpoint)

        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(filename):
            os.remove(filename)

async def generate_tts(sentence: str, voice: str, speed: str, client_session: dict):
    logging.info(f"Generating TTS with voice: {voice}, speed: {speed}")
    
    tts_config = client_session.get('TTS-Config', {})
    base_url = tts_config.get('TTS_SERVER_URL', 'http://213.173.96.19')

    tts_output = text_to_speech(sentence, voice, speed, base_url)
    if tts_output is None:
        logging.error(f"TTS generation failed for sentence: {sentence}")
        return None
    
    tts_filename = f"tts_output_{hash(sentence)}.wav"
    with open(tts_filename, 'wb') as tts_file:
        tts_file.write(tts_output)
    return tts_filename



@app.post("/generate_tts")
async def generate_tts_endpoint(request: SentenceRequest):
    client_id = request.client_id
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    client_session = client_sessions[client_id]
    
    logger.info(f"Received TTS request for sentence: {request.sentence}")
    logger.info(f"Initial Voice: {request.voice}, Speed: {request.speed}")
    
    tts_filename = await generate_tts(request.sentence, request.voice, request.speed, client_session)

    if tts_filename is None:
        logger.error("TTS generation failed")
        raise HTTPException(status_code=500, detail="TTS generation failed for all voices")
    
    return FileResponse(tts_filename, media_type="audio/wav", filename=tts_filename)



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
    print(config)

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
                    response = requests.get(f"http://90.186.125.172:8002/send_file?client_id={client_id}&file={file_path}")
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

