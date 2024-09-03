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
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.websockets import WebSocketDisconnect
import json
import time
import os
import logging
import asyncio
import io
import wave
import pyaudio
import requests
import nltk
from nltk.tokenize import sent_tokenize
import traceback

# Ensure NLTK punkt tokenizer is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


import sys
from types import ModuleType
import json
import re
import base64
import pandas as pd
import asyncio
import time
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from typing import Optional
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from fastapi import Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from fastapi.responses import JSONResponse
from bud_e_llm import ask_LLM
from bud_e_captioning_with_ocr import send_image_for_captioning_and_ocr, analyze_clipboard


# Set up global DataFrame
api_df = pd.DataFrame(columns=['api_key', 'requests_last_hour', 'email'])
api_df.loc[len(api_df)] = ['12345', 0, 'dummy@example.com']  # Add dummy API key
api_df['api_key'] = api_df['api_key'].astype(str)

print(api_df)
# File to save DataFrame
DF_FILE = 'api_usage.csv'

# Load existing data if file exists
if os.path.exists(DF_FILE):
    api_df = pd.read_csv(DF_FILE)

# API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key")


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


logger = logging.getLogger(__name__)


app = FastAPI()

client_sessions = {}
 
is_running = True



class ClientConfig(BaseModel):
    LLM_Config: Dict[str, Any]
    TTS_Config: Dict[str, Any]
    Skills: str
    Conversation_History: List[Dict[str, str]]
    Scratchpad: Dict[str, Any]
    System_Prompt: str
    API_Key: str
    client_id: Optional[str] = None
    clipboard: Optional[str] = None  
    screen: Optional[bytes] = None   
    code_for_client_execution: Optional[str] = None  


async def save_df_periodically():
    while True:
        api_df.to_csv(DF_FILE, index=False)
        print(f"DataFrame saved to {DF_FILE}")
        await asyncio.sleep(60)  # Save every minute

# Start the periodic saving task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(save_df_periodically())



# Update the verify_api_key function
def verify_api_key(api_key: str = Depends(api_key_header)):
    logger.debug(f"Verifying API key: {api_key}")
    
    # Convert all keys to strings and lowercase for case-insensitive comparison
    valid_keys = set(api_df['api_key'].astype(str).str.lower())
    
    logger.debug(f"Valid keys: {valid_keys}")
    
    if api_key.lower() in valid_keys:
        logger.info(f"API key {api_key} verified successfully")
        return api_key
    else:
        logger.warning(f"Invalid API key: {api_key}")
        raise HTTPException(status_code=403, detail="Invalid API key")



class AskLLMRequest(BaseModel):
    client_session: ClientConfig
    user_input: str

@app.post("/ask_llm")
async def ask_llm_endpoint(request: AskLLMRequest, api_key: str = Depends(verify_api_key)):
    client_session = request.client_session.dict()
    user_input = request.user_input

    if client_session['client_id'] not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    try:
        # Prepare the conversation history
        conversation_history = client_session['Conversation_History']
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

        # Get LLM configuration
        llm_config = client_session['LLM_Config']

        # Call the ask_LLM function
        ai_response = ask_LLM(
            llm_config['model'],
            client_session['System_Prompt'],
            conversation_history_str,
            temperature=llm_config['temperature'],
            top_p=llm_config['top_p'],
            max_tokens=llm_config['max_tokens'],
            frequency_penalty=llm_config['frequency_penalty'],
            presence_penalty=llm_config['presence_penalty'],
            streaming=False
        )

        # Update the conversation history
        conversation_history.append({"role": "assistant", "content": ai_response})
        client_session['Conversation_History'] = conversation_history

        # Update the client_sessions dictionary
        client_sessions[client_session['client_id']] = client_session

        # Prepare the response
        response_data = {
            "ai_response": ai_response,
            "updated_client_session": client_session
        }

        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.error(f"An error occurred in ask_llm_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



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
    #print(skills_code)
    lm_activated_skills = parse_lm_activated_skills(skills_code)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #print("lm_activated_skills:", lm_activated_skills)
    
    # Import the skills code
    imported_functions = import_skills_code(skills_code)
    print("Successfully imported functions:", ", ".join(imported_functions.keys()))
    
    skill_calls = extract_skill_calls(ai_response, lm_activated_skills)
    print("skill_calls:", skill_calls)
    
    updated_config = client_session.copy()
    skill_response=""
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
    
    return skill_response, updated_config

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
async def generate_client_id_endpoint(request: Request, api_key: str = Depends(verify_api_key)):
    logging.info(f"Received client ID generation request with API key: {api_key}")
    
    try:
        # Parse the raw request body
        raw_data = await request.json()
        
        # Validate the data against the ClientConfig model
        config = ClientConfig(**raw_data)
    except ValidationError as e:
        logging.error(f"Invalid client configuration: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing client configuration: {e}")
        raise HTTPException(status_code=400, detail="Invalid request data")

    new_client_id = generate_client_id()
    logging.info(f"Generated new client ID: {new_client_id}")

    # Create a new dictionary with all fields from config plus the new client_id
    client_session = config.dict()
    client_session['client_id'] = new_client_id

    client_sessions[new_client_id] = client_session
    
    return JSONResponse(content={"client_id": new_client_id}, status_code=200)



@app.post("/receive_audio")
async def receive_audio(
    client_id: str = Form(...),
    file: UploadFile = File(...),
    client_config: str = Form(...),
    api_key: str = Depends(verify_api_key)
):
    logging.info(f"Received audio from client: {client_id}")
    #logging.debug(f"Client config: {client_config}")
    logging.debug(f"API Key: {api_key}")
    #print(client_sessions)
    
    """
    Receive audio file, process it, and return the response.
    
    Parameters:
    - client_id: ID of the client sending the request
    - file: Audio file to be processed
    - client_config: Client configuration in JSON format
    - api_key: API key for authentication (injected by dependency)
    
    Returns:
    - JSON response with processed data and audio
    """
    startendpoint = time.time()
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    # Parse the client_config JSON string
    try:
        client_session = json.loads(client_config)
        config = ClientConfig(**client_session)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in client_session")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid data in client_session")
    print("#############################12345")
    # ###### skill_result =await execute_skill_on_client(client_session=client_session) # TEST
    client_sessions[client_id]=client_session
    client_sessions[client_id]["code_for_client_execution"] = ""
    skills_code = client_sessions[client_id].get('Skills', '')
    lm_activated_skills = parse_lm_activated_skills(skills_code)
    logging.info(f"Parsed LM activated skills: {lm_activated_skills}")
    
    filename = f"{client_id}_audio.wav"
    with open(filename, "wb") as audio_file:
        audio_file.write(await file.read())

    try:
        start_time = time.time()
        user_input = transcribe(filename, client_id)
        end_time = time.time()
        print(f"ASR Time: {end_time - start_time} seconds")
        print(f"User: {user_input}")
        #updated_transcription =user_input #+ analyze_clipboard(client_session)
        print("+++++++++++++++++++++")
        #print(updated_transcription)

        #start_time = time.time()
        #language = detect_language(user_input)
        #end_time = time.time()
        #print(f"Language Detected: {language}")
        #print(f"Language Detection Time: {end_time - start_time} seconds")

        keyword_activated_skills = extract_activated_skills_from_code([skills_code])

        skills = []

        skill_response = ""
        for skill_name, skill_comment in keyword_activated_skills.items():
            conditions_list = parse_list_of_lists(skill_comment)

            if any(all(cond.lower() in user_input.lower() for cond in condition) for condition in conditions_list):
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                skills.append(skill_name)


        for skill_name in skills:
            skill_response, client_session = skill_execution(
                skill_name, user_input, client_sessions[client_id])
            # Update the client_sessions dictionary with the new session data
            client_sessions[client_session['client_id']] = client_session
            

        system_prompt = client_sessions[client_id].get('System_Prompt', '')
        llm_config = client_sessions[client_id].get('LLM_Config', {})
        print("LLM CONFIG:", llm_config)

        conversation_history = client_sessions[client_id].get('Conversation_History', [])
        if skill_response != "":
            print("SKILL RESPONSE:" ,skill_response)
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": skill_response})
        else:
            conversation_history.append({"role": "user", "content": user_input})

        conversation_history_str = ""
        for message in conversation_history:
            if message["role"] == "user":
                conversation_history_str += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                conversation_history_str += f"BUD-E: {message['content']}\n"

        start_time = time.time()
        ai_response = ""
        first_sentence = ""
        rest_of_text = ""
        first_sentence_complete = False
        sentences = []
        tts_task = None

        streaming = llm_config.get('streaming', True)

        if streaming:
            for chunk in ask_LLM(
                llm_config['model'],
                system_prompt,
                conversation_history_str,
                temperature=llm_config['temperature'],
                top_p=llm_config['top_p'],
                max_tokens=llm_config['max_tokens'],
                frequency_penalty=llm_config['frequency_penalty'],
                presence_penalty=llm_config['presence_penalty'],
                streaming=True
            ):
                if chunk:
                    ai_response += chunk
                    if not first_sentence_complete:
                        first_sentence += chunk
                        if len(first_sentence) >10 and any(char in first_sentence for char in ['. ', '!', '?']):  #### conflicts with urls in skill calls
                            first_sentence_complete = True
                            if "</" in first_sentence and ">" in first_sentence: 
                               # Process LM activated skills on the full AI response
                               first_sentence, client_sessions[client_id] = process_lm_activated_skills(first_sentence, client_sessions[client_id], user_input)
                               print("STREAMING & FIRST SENTENCE COMPLETE, first_sentence_test:" , first_sentence)
    
                            sentences.append(first_sentence)
                            
                            # Start TTS generation for the first sentence
                            tts_config = client_sessions[client_id].get('TTS_Config', {})
                            voice = tts_config.get('voice', 'Stefanie')
                            speed = tts_config.get('speed', 'normal')
                            logger.info(f"Attempting TTS generation with voice: {voice}, speed: {speed}")
                            tts_task = asyncio.create_task(generate_tts(first_sentence, voice, speed, client_sessions[client_id]))
                    else:
                        rest_of_text += chunk
        else:
            # Non-streaming mode
            ai_response = ask_LLM(
                llm_config['model'],
                system_prompt,
                conversation_history_str,
                temperature=llm_config['temperature'],
                top_p=llm_config['top_p'],
                max_tokens=llm_config['max_tokens'],
                frequency_penalty=llm_config['frequency_penalty'],
                presence_penalty=llm_config['presence_penalty'],
                streaming=False
            )
            sentences = re.split(r'(?<=[.!?])\s+', ai_response)
            first_sentence = sentences[0]
            rest_of_text = ' '.join(sentences[1:])

            if "</" in first_sentence and ">" in first_sentence: 
              # Process LM activated skills on the full AI response
              first_sentence, client_sessions[client_id] = process_lm_activated_skills(first_sentence, client_sessions[client_id], user_input)
              print("NON STREAMING & FIRST SENTENCE COMPLETE, first_sentence_test:", first_sentence)


            # Start TTS generation for the first sentence
            tts_config = client_sessions[client_id].get('TTS_Config', {})
            voice = tts_config.get('voice', 'Stefanie')
            speed = tts_config.get('speed', 'normal')
            logger.info(f"Attempting TTS generation with voice: {voice}, speed: {speed}")
            tts_task = asyncio.create_task(generate_tts(first_sentence, voice, speed, client_sessions[client_id]))

        # Handle case where there's no sentence-ending punctuation in streaming mode
        if streaming and not first_sentence_complete:
            first_sentence = ai_response.strip()
            if "</" in first_sentence and ">" in first_sentence: 
              # Process LM activated skills on the full AI response
              first_sentence, client_sessions[client_id] = process_lm_activated_skills(first_sentence, client_sessions[client_id], user_input)
              print("STREAMING & FIRST SENTENCE NOT COMPLETE, first_sentence_test:" , first_sentence)


            sentences = [first_sentence]
            rest_of_text = ""

            # Start TTS generation for the first sentence if not already started
            if not tts_task:
                tts_config = client_sessions[client_id].get('TTS_Config', {})
                voice = tts_config.get('voice', 'Stefanie')
                speed = tts_config.get('speed', 'normal')
                logger.info(f"Attempting TTS generation with voice: {voice}, speed: {speed}")
                tts_task = asyncio.create_task(generate_tts(first_sentence, voice, speed, client_sessions[client_id]))

        end_time = time.time()
        print(f"LLM Time: {end_time - start_time} seconds")
        print(f"AI: {ai_response}")

        # Split the rest of the text into sentences and add to the sentences list
        if rest_of_text:
            sentences.extend(sent_tokenize(rest_of_text.strip()))

        ai_response = " ".join(sentences)
        conversation_history.append({"role": "assistant", "content": ai_response})


        # Prepare the response data
        response_data = {
            "sentences": sentences[1:],
            "updated_conversation_history": conversation_history,
            "config_updates": client_sessions[client_id]
        }

        # Convert JSON data to bytes and add a separator
        json_bytes = json.dumps(response_data).encode('utf-8')
        separator = b'\n---AUDIO_DATA---\n'

        # Wait for the TTS task to complete
        tts_filename = await tts_task if tts_task else None

        if tts_filename is None:
            logger.error("TTS generation failed for the first sentence")
            # Instead of raising an exception, we'll return an error message in the response
            error_message = "TTS generation failed for the first sentence. Proceeding with text-only response."
            logger.warning(error_message)
            
            # Add error message to the response data
            response_data["error"] = error_message

            # Return JSON response without audio data
            return Response(content=json.dumps(response_data).encode('utf-8'), media_type="application/json")

        # Read the audio file
        with open(tts_filename, 'rb') as audio_file:
            first_sentence_audio = audio_file.read()

        # Combine JSON data and audio data
        combined_data = json_bytes + separator + first_sentence_audio

        print("TIME FULL ENDPOINT SCRIPT:", time.time() - startendpoint)

        # Clean up the temporary audio file
        os.remove(tts_filename)

        return Response(content=combined_data, media_type="application/octet-stream")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(content=json.dumps({"error": str(e)}).encode('utf-8'), 
                        status_code=500, 
                        media_type="application/json")

    finally:
        if os.path.exists(filename):
            os.remove(filename)

async def generate_tts(sentence: str, voice: str, speed: str, client_session: dict):
    logging.info(f"Generating TTS with voice: {voice}, speed: {speed}")
    
    tts_config = client_session.get('TTS-Config', {})
    base_url = tts_config.get('TTS_SERVER_URL', 'http://213.173.96.19')
    print("DEBUG:",sentence, voice, speed, base_url)
    tts_output = text_to_speech(sentence, voice, speed, base_url)
    if tts_output is None:
        
        logging.error(f"TTS generation failed for sentence: {sentence}")
        return None
    
    tts_filename = f"tts_output_{hash(sentence)}.wav"
    with open(tts_filename, 'wb') as tts_file:
        tts_file.write(tts_output)
    return tts_filename



class SentenceRequest(BaseModel):
    sentence: str
    client_id: str
    voice: Optional[str] = "Stefanie"
    speed: Optional[str] = "normal"

@app.post("/generate_tts")
async def generate_tts_endpoint(request: SentenceRequest, api_key: str = Depends(verify_api_key)):
    """
    Generate text-to-speech audio for a given sentence.
    
    Parameters:
    - request: SentenceRequest object containing sentence, client_id, and optional voice and speed
    - api_key: API key for authentication (injected by dependency)
    
    Returns:
    - Audio file response
    """
    client_id = request.client_id
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    client_session = client_sessions[client_id]
    
    logger.info(f"Received TTS request for sentence: {request.sentence}")
    logger.info(f"Voice: {request.voice}, Speed: {request.speed}")
    
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


@app.post("/update_client_config/{client_id}")
async def update_client_config(client_id: str, config: dict, api_key: str = Depends(verify_api_key)):
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    client_sessions[client_id].update(config)
    print(config)

    return JSONResponse(content={"message": "Client configuration updated successfully"}, status_code=200)

@app.get("/get_client_data/{client_id}")
async def get_client_data_endpoint(client_id: str, api_key: str = Depends(verify_api_key)):

    if client_id not in client_sessions:
        raise HTTPException(status_code=404, detail="Client not found")
    return JSONResponse(content=client_sessions[client_id], status_code=200)


def print_menu():
    """Function to print the server menu."""
    print("1. Exit")
    print("Enter your choice (1): ", end="", flush=True)

def run_cli():
    """Function to run the command-line interface."""
    global is_running
    while is_running:
        print_menu()
        choice = input()
        if choice == "1":
            is_running = False
            break
        else:
            print("Invalid choice. Please try again.")

def run_server():
    """Function to run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8001)

def cleanup_old_tts_files():
    while True:
        current_time = time.time()
        for filename in os.listdir('.'):
            if filename.startswith('tts_output') and filename.endswith('.wav'):
                file_path = os.path.join('.', filename)
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 60:  # 60 seconds = 1 minute
                    try:
                        os.remove(file_path)
                        print(f"Deleted old file: {filename}")
                    except Exception as e:
                        print(f"Error deleting {filename}: {e}")
        time.sleep(60)  # Sleep for 60 seconds before the next cleanup


if __name__ == "__main__":

    # Start the FastAPI server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Start the cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_tts_files, daemon=True)
    cleanup_thread.start()


    # Run the CLI in the main thread
    run_cli()

    print("Shutting down...")

