# Import standard libraries
import asyncio  # For asynchronous operations
import os  # For interacting with the operating system
import sys  # System-specific parameters and functions
import threading  # For running threads
import time  # Time-related functions
import logging  # Logging facilities
import hashlib  # Secure hash algorithms and message digests
import mimetypes  # Map filenames to MIME types
import re  # Regular expressions
import base64  # Base64 encoding and decoding
import io  # Core tools for working with streams
import traceback  # For extracting and printing stack traces
from types import ModuleType  # For creating new modules dynamically

# Import third-party libraries
import pyaudio  # For audio input/output
import wave  # For reading and writing WAV files
import requests  # For making HTTP requests
import pandas as pd  # For data manipulation and analysis
import nltk  # Natural Language Toolkit
from nltk.tokenize import sent_tokenize  # Sentence tokenizer
import json

from pydantic import BaseModel, ValidationError  # For data validation using Python type annotations
from typing import List, Dict, Any, Optional  # For type annotations
from fastapi import (
    FastAPI, HTTPException, Depends, Request, Form,
    File, UploadFile, Query
)  # For building APIs with Python
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse  # For returning responses
from fastapi.security import APIKeyHeader  # For API key authentication
import uvicorn  # For serving FastAPI applications

# Import custom modules
from bud_e_transcribe import transcribe  # Custom module for speech-to-text transcription
from bud_e_tts import text_to_speech  # Custom module for text-to-speech conversion
from bud_e_llm import ask_LLM  # Custom module for interacting with a Language Model
from bud_e_captioning_with_ocr import send_image_for_captioning_and_ocr, analyze_clipboard  # Custom modules for image captioning and analysis

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK 'punkt' tokenizer is downloaded for sentence tokenization
nltk.download('punkt', quiet=True)

# Create a FastAPI application instance
app = FastAPI()

# Global dictionary to store client sessions
client_sessions = {}

# Flag to indicate if the server is running (used in the CLI)
is_running = True

# Pydantic model representing the client configuration
class ClientConfig(BaseModel):
    LLM_Config: Dict[str, Any]  # Configuration for the Language Model
    TTS_Config: Dict[str, Any]  # Configuration for Text-to-Speech
    Skills: str  # Skills code provided by the client
    Conversation_History: List[Dict[str, str]]  # Conversation history between user and assistant
    Scratchpad: Dict[str, Any]  # Scratchpad for temporary data
    System_Prompt: str  # System prompt for the assistant
    API_Key: str  # API key for authentication
    client_id: Optional[str] = None  # Optional client ID
    clipboard: Optional[str] = None  # Optional clipboard content
    screen: Optional[bytes] = None  # Optional screen content
    code_for_client_execution: Optional[str] = None  # Code for execution on the client side

# Global DataFrame to store API keys and usage data
api_df = pd.DataFrame(columns=['api_key', 'requests_last_hour', 'email'])
api_df.loc[len(api_df)] = ['12345', 0, 'dummy@example.com']  # Add a dummy API key
api_df['api_key'] = api_df['api_key'].astype(str)  # Ensure API keys are strings

# File to save the API usage DataFrame
DF_FILE = 'api_usage.csv'

# Load existing data if the DataFrame file exists
if os.path.exists(DF_FILE):
    api_df = pd.read_csv(DF_FILE)

# APIKeyHeader dependency for API key authentication
api_key_header = APIKeyHeader(name="X-API-Key")

# -----------------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------------

# Function to periodically save the API usage DataFrame to a CSV file
async def save_df_periodically():
    while True:
        # Save the DataFrame to a CSV file
        api_df.to_csv(DF_FILE, index=False)
        print(f"DataFrame saved to {DF_FILE}")
        # Wait for 60 seconds before saving again
        await asyncio.sleep(60)  # Save every minute

# Event handler to start the periodic saving task on application startup
@app.on_event("startup")
async def startup_event():
    # Create a background task to save the DataFrame periodically
    asyncio.create_task(save_df_periodically())

# Function to verify the API key provided in the request header
def verify_api_key(api_key: str = Depends(api_key_header)):
    # Convert all valid API keys to lowercase for case-insensitive comparison
    valid_keys = set(api_df['api_key'].astype(str).str.lower())
    
    # Check if the provided API key is in the set of valid keys
    if api_key.lower() in valid_keys:
        logger.info(f"API key {api_key} verified successfully")
        return api_key  # Return the API key if verification is successful
    else:
        logger.warning(f"Invalid API key: {api_key}")
        # Raise an HTTPException if the API key is invalid
        raise HTTPException(status_code=403, detail="Invalid API key")

# Pydantic model for the request to the /ask_llm endpoint
class AskLLMRequest(BaseModel):
    client_session: ClientConfig  # Client configuration
    user_input: str  # User input text

# Endpoint to handle requests to the /ask_llm route
@app.post("/ask_llm")
async def ask_llm_endpoint(request: AskLLMRequest, api_key: str = Depends(verify_api_key)):
    """
    Endpoint to process a request to ask the Language Model (LLM).

    Parameters:
    - request: AskLLMRequest object containing client session and user input
    - api_key: API key for authentication (provided by dependency)

    Returns:
    - JSONResponse with the AI's response and updated client session
    """
    # Extract client session and user input from the request
    client_session = request.client_session.dict()
    user_input = request.user_input

    # Check if the client ID exists in the client_sessions dictionary
    if client_session['client_id'] not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    try:
        # Prepare the conversation history by appending the user input
        conversation_history = client_session['Conversation_History']
        conversation_history.append({"role": "user", "content": user_input})
        
        # Convert conversation history to a string format suitable for the LLM
        conversation_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

        # Get LLM configuration from the client session
        llm_config = client_session['LLM_Config']

        # Call the ask_LLM function to get the AI's response
        ai_response = ask_LLM(
            model=llm_config['model'],
            system_prompt=client_session['System_Prompt'],
            conversation_history=conversation_history_str,
            temperature=llm_config['temperature'],
            top_p=llm_config['top_p'],
            max_tokens=llm_config['max_tokens'],
            frequency_penalty=llm_config['frequency_penalty'],
            presence_penalty=llm_config['presence_penalty'],
            streaming=False  # Set streaming to False for synchronous response
        )

        # Update the conversation history with the AI's response
        conversation_history.append({"role": "assistant", "content": ai_response})
        client_session['Conversation_History'] = conversation_history

        # Update the client_sessions dictionary with the updated client session
        client_sessions[client_session['client_id']] = client_session

        # Prepare the response data
        response_data = {
            "ai_response": ai_response,
            "updated_client_session": client_session
        }

        # Return the response as JSON
        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        logger.error(f"An error occurred in ask_llm_endpoint: {str(e)}")
        # Return an HTTP 500 error if an exception occurs
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Function to generate a random client ID
def generate_client_id():
    """
    Generate a random client ID using SHA-256 hash of random bytes.

    Returns:
    - A 10-character hexadecimal client ID
    """
    # Generate random bytes and compute SHA-256 hash, take the first 10 characters
    return hashlib.sha256(os.urandom(10)).hexdigest()[:10]

# Function to parse LM ACTIVATED SKILL comments and function definitions from code
def parse_lm_activated_skills(code_string: str) -> dict:
    """
    Parse code to find LM ACTIVATED SKILLs along with their opening and closing tags.

    Parameters:
    - code_string: String containing the code to parse

    Returns:
    - Dictionary mapping function names to their tag information
    """
    # Regular expression pattern to match LM ACTIVATED SKILL comments and function definitions
    pattern = r'# LM ACTIVATED SKILL:.*?USAGE INSTRUCTIONS:.*?<(\w+)>(.*?)</\1>.*?\n(def\s+(\w+)\(.*?\):.*?)\n\n'
    
    # Find all matches in the code string using the regular expression
    matches = re.findall(pattern, code_string, re.DOTALL)
    
    # Dictionary to store the extracted skills
    skills = {}
    
    # Process each match found in the code
    for match in matches:
        tag_name, _, function_def, function_name = match
        opening_tag = f"<{tag_name}>"
        closing_tag = f"</{tag_name}>"
        
        # Add the skill information to the skills dictionary
        skills[function_name] = {
            "opening_tag": opening_tag,
            "closing_tag": closing_tag,
            'function_name': function_name
        }
    
    # Return the dictionary of skills
    return skills

# Function to extract skill calls from the AI's response based on LM ACTIVATED SKILL tags
def extract_skill_calls(ai_response: str, lm_activated_skills: dict) -> List[dict]:
    """
    Extract skill calls from the AI response text.

    Parameters:
    - ai_response: The AI's response text
    - lm_activated_skills: Dictionary of LM activated skills with their tags

    Returns:
    - List of dictionaries containing skill call information
    """
    # List to store extracted skill calls
    skill_calls = []
    
    # Iterate over each skill in the activated skills
    for skill_title, skill_info in lm_activated_skills.items():
        opening_tag = re.escape(skill_info['opening_tag'])
        closing_tag = re.escape(skill_info['closing_tag'])
        # Regular expression pattern to find content between opening and closing tags
        pattern = rf'{opening_tag}(.*?){closing_tag}'
        # Find all matches in the AI response
        matches = re.findall(pattern, ai_response, re.DOTALL)
        for content in matches:
            # Append the skill call information to the list
            skill_calls.append({
                "name": skill_title,
                "function_name": skill_info['function_name'],
                "parameters": content.strip()
            })
    # Return the list of skill calls
    return skill_calls

# Function to import skills code dynamically into the module namespace
def import_skills_code(skills_code: str) -> dict:
    """
    Import skills code provided as a string and extract callable functions.

    Parameters:
    - skills_code: String containing the Python code defining skills

    Returns:
    - Dictionary mapping function names to function objects
    """
    # Create a new module to hold the skills
    skills_module = ModuleType("dynamic_skills")
    
    # Add the module to sys.modules so it can be imported
    sys.modules["dynamic_skills"] = skills_module
    
    # Dictionary to hold successfully imported functions
    imported_functions = {}
    
    try:
        # Execute the skills code in the context of the new module
        exec(skills_code, skills_module.__dict__)
        
        # Iterate through the module's attributes
        for attr_name in dir(skills_module):
            attr = getattr(skills_module, attr_name)
            # Check if it's a function and doesn't start with an underscore
            if callable(attr) and not attr_name.startswith("_"):
                # Add the function to the imported functions dictionary
                imported_functions[attr_name] = attr
                    
        print(f"Successfully imported functions: {', '.join(imported_functions.keys())}")
    except Exception as e:
        print(f"Error importing skills code: {str(e)}")
        print(traceback.format_exc())
    
    # Return the dictionary of imported functions
    return imported_functions

# Function to process LM activated skills within the AI's response
def process_lm_activated_skills(ai_response: str, client_session: dict, user_input: str) -> (str, dict):
    """
    Process LM ACTIVATED SKILLs found in the AI response by executing corresponding functions.

    Parameters:
    - ai_response: The AI's response text
    - client_session: The client session data
    - user_input: The user's input text

    Returns:
    - Tuple containing the modified AI response and updated client session
    """
    # Get the skills code from the client session
    skills_code = client_session.get('Skills', '')
    
    # Parse the code to find LM ACTIVATED SKILLs
    lm_activated_skills = parse_lm_activated_skills(skills_code)
    #print("lm_activated_skills:", lm_activated_skills)
    
    # Import the skills code to obtain callable functions
    imported_functions = import_skills_code(skills_code)
    #print("Successfully imported functions:", ", ".join(imported_functions.keys()))
    
    # Extract skill calls from the AI response
    skill_calls = extract_skill_calls(ai_response, lm_activated_skills)
    #print("skill_calls:", skill_calls)
    
    # Copy of the client session to work with
    updated_config = client_session.copy()
    
    # Initialize skill response
    skill_response = ""
    
    # Iterate over each skill call
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
                        # Ensure params are correctly formatted
                        formatted_params = f"({params})" if not params.startswith("(") else params
                        # Execute the skill function
                        skill_response, updated_config = skill_function(user_input, updated_config, formatted_params)
                        
                        # Replace the skill call in the AI response with the skill response
                        ai_response = re.sub(
                            rf'{opening_tag}.*?{closing_tag}',
                            skill_response,
                            ai_response,
                            count=1
                        )
                        #print(f"{skill_name} executed. Response: {skill_response}")
                    except Exception as e:
                        print(f"Error executing function {function_name}: {str(e)}")
                        print(traceback.format_exc())
                else:
                    print(f"Warning: Function {function_name} not found in imported functions.")
            else:
                print(f"Warning: Skill call for {skill_name} found, but not properly enclosed in tags in AI response.")
        else:
            print(f"Warning: Skill info for {skill_name} not found in lm_activated_skills.")
        
    # Return the modified AI response and updated client session
    return ai_response, updated_config

# Function to execute a skill function with provided parameters
def skill_execution(function_name: str, transcription_response: str, client_session: dict, LMGeneratedParameters: str = "") -> (str, dict):
    """
    Execute a skill function based on its name and parameters.

    Parameters:
    - function_name: Name of the function to execute
    - transcription_response: The transcribed user input
    - client_session: The client session data
    - LMGeneratedParameters: Parameters generated by the Language Model

    Returns:
    - Tuple containing the skill's response and updated client session
    """
    # Print that we are executing the skill
    print(f"Executing skill: {function_name}")
    
    # Get the skills code from the client session
    skills_code = client_session.get('Skills', '')
    
    # Create a module from the skills code
    module = {}
    exec(skills_code, module)
    
    # Try to get the function from the dynamically created module
    function_to_run = module.get(function_name)
    
    # Check if the function exists
    if function_to_run is None:
        raise ValueError(f"The specified function '{function_name}' is not defined.")
    
    # Check if the function is callable
    if not callable(function_to_run):
        raise ValueError(f"The function '{function_name}' is not callable.")
    
    # Execute the function
    skill_response, updated_client_session = function_to_run(
        transcription_response, client_session, LMGeneratedParameters
    )
    
    # Return the skill's response and updated client session
    return skill_response, updated_client_session

# Function to parse a string representation of a list of lists
def parse_list_of_lists(input_str: str) -> List[List[str]]:
    """
    Parses a string representing a list of lists, where each sublist contains strings.

    Parameters:
    - input_str: String representation of a list of lists

    Returns:
    - List of lists of strings
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

    # Return the parsed list of lists
    return result

# Function to extract activated skills from code based on a keyword
def extract_activated_skills_from_code(code_strings: List[str], keyword: str = "KEYWORD ACTIVATED SKILL:") -> dict:
    """
    Extract functions with a comment containing a specific keyword.

    Parameters:
    - code_strings: List of code strings to search
    - keyword: The keyword to look for in comments (default: "KEYWORD ACTIVATED SKILL:")

    Returns:
    - Dictionary mapping function names to associated comments
    """
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

# -----------------------------------------------------------------------------
# Endpoint Definitions
# -----------------------------------------------------------------------------

# Root endpoint to check if the server is running
@app.get("/")
async def root():
    """
    Root endpoint to verify if the server is running.

    Returns:
    - JSON message indicating the server is running
    """
    return {"message": "Server is running"}

# Endpoint to generate a new client ID
@app.post("/generate_client_id")
async def generate_client_id_endpoint(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Endpoint to generate a new client ID.

    Parameters:
    - request: The incoming HTTP request
    - api_key: API key for authentication (provided by dependency)

    Returns:
    - JSONResponse containing the new client ID
    """
    logging.info(f"Received client ID generation request with API key: {api_key}")
    
    try:
        # Parse the raw request body as JSON
        raw_data = await request.json()
        
        # Validate the data against the ClientConfig model
        config = ClientConfig(**raw_data)
    except ValidationError as e:
        logging.error(f"Invalid client configuration: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing client configuration: {e}")
        raise HTTPException(status_code=400, detail="Invalid request data")

    # Generate a new client ID
    new_client_id = generate_client_id()
    logging.info(f"Generated new client ID: {new_client_id}")

    # Create a new dictionary with all fields from config plus the new client_id
    client_session = config.dict()
    client_session['client_id'] = new_client_id

    # Store the client session in the global client_sessions dictionary
    client_sessions[new_client_id] = client_session
    
    # Return the new client ID as a JSON response
    return JSONResponse(content={"client_id": new_client_id}, status_code=200)

# Endpoint to receive audio, process it, and return the response
@app.post("/receive_audio")
async def receive_audio(
    client_id: str = Form(...),
    file: UploadFile = File(...),
    client_config: str = Form(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Receive audio file, process it, and return the response.

    Parameters:
    - client_id: ID of the client sending the request
    - file: Audio file to be processed
    - client_config: Client configuration in JSON format
    - api_key: API key for authentication (provided by dependency)

    Returns:
    - Response containing JSON data and audio data
    """
    logging.info(f"Received audio from client: {client_id}")
    # Start timer for measuring processing time
    start_endpoint = time.time()

    # Check if the client ID exists in the client_sessions dictionary
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    # Parse the client_config JSON string
    try:
        client_session = json.loads(client_config)
        # Validate the client session using ClientConfig model
        config = ClientConfig(**client_session)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in client_session")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid data in client_session")

    # Update the client_sessions dictionary with the latest client session
    client_sessions[client_id] = client_session
    client_sessions[client_id]["code_for_client_execution"] = ""

    # Get the skills code from the client session
    skills_code = client_sessions[client_id].get('Skills', '')
    # Parse LM activated skills from the skills code
    lm_activated_skills = parse_lm_activated_skills(skills_code)
    logging.info(f"Parsed LM activated skills: {lm_activated_skills}")
    
    # Save the uploaded audio file temporarily
    filename = f"{client_id}_audio.wav"
    with open(filename, "wb") as audio_file:
        audio_file.write(await file.read())

    try:
        # Start timer for transcription
        start_time = time.time()
        # Transcribe the audio file to text
        user_input = transcribe(filename, client_id)
        end_time = time.time()
        print(f"ASR Time: {end_time - start_time} seconds")
        print(f"User: {user_input}")

        # Extract keyword activated skills from the skills code
        keyword_activated_skills = extract_activated_skills_from_code([skills_code])
        skills = []
        skill_response = ""

        # Check if any skills are triggered by the user's input
        for skill_name, skill_comment in keyword_activated_skills.items():
            conditions_list = parse_list_of_lists(skill_comment)
            # Check if any condition matches the user input
            if any(all(cond.lower() in user_input.lower() for cond in condition) for condition in conditions_list):
                skills.append(skill_name)

        # Execute the triggered skills
        for skill_name in skills:
            skill_response, client_session = skill_execution(
                skill_name, user_input, client_sessions[client_id])
            # Update the client_sessions dictionary with the new session data
            client_sessions[client_session['client_id']] = client_session

        # Prepare conversation history and LLM configuration
        system_prompt = client_sessions[client_id].get('System_Prompt', '')
        llm_config = client_sessions[client_id].get('LLM_Config', {})
        print("LLM CONFIG:", llm_config)

        conversation_history = client_sessions[client_id].get('Conversation_History', [])
        if skill_response != "":
            # If a skill response exists, add it to the conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": skill_response})
        else:
            # Otherwise, just add the user input
            conversation_history.append({"role": "user", "content": user_input})

        # Convert conversation history to string format
        conversation_history_str = ""
        for message in conversation_history:
            if message["role"] == "user":
                conversation_history_str += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                conversation_history_str += f"BUD-E: {message['content']}\n"

        # Start timer for LLM processing
        start_time = time.time()
        ai_response = ""
        first_sentence = ""
        rest_of_text = ""
        first_sentence_complete = False
        sentences = []
        tts_task = None

        # Check if streaming is enabled in LLM configuration
        streaming = llm_config.get('streaming', True)
        if streaming:
            # Streaming mode: process LLM response in chunks
            for chunk in ask_LLM(
                modelname=llm_config['model'],
                systemprompt=system_prompt,
                content=conversation_history_str,
                temperature=llm_config['temperature'],
                top_p=llm_config['top_p'],
                max_tokens=llm_config['max_tokens'],
                frequency_penalty=llm_config['frequency_penalty'],
                presence_penalty=llm_config['presence_penalty'],
                streaming=True
            ):
                if chunk:
                    # Append chunk to the AI response
                    ai_response += chunk
                    if not first_sentence_complete:
                        # Build the first sentence until a sentence-ending punctuation is found
                        first_sentence += chunk
                        if len(first_sentence) > 10 and any(char in first_sentence for char in ['. ', '! ', '? ']):
                            first_sentence_complete = True

                            # Process LM activated skills on the first sentence if applicable
                            if "</" in first_sentence and ">" in first_sentence:
                                first_sentence, client_sessions[client_id] = process_lm_activated_skills(first_sentence, client_sessions[client_id], user_input)
                            
                            # Add the first sentence to the sentences list
                            sentences.append(first_sentence)
                            
                            # Start TTS generation for the first sentence
                            tts_config = client_sessions[client_id].get('TTS_Config', {})
                            voice = tts_config.get('voice', 'Stefanie')
                            speed = tts_config.get('speed', 'normal')
                            tts_task = asyncio.create_task(generate_tts(first_sentence, voice, speed, client_sessions[client_id]))
                    else:
                        # Build the rest of the text
                        rest_of_text += chunk

                    # Process LM activated skills on the rest of the text if applicable
                    if "</" in rest_of_text and ">" in rest_of_text:
                        rest_of_text, client_sessions[client_id] = process_lm_activated_skills(rest_of_text, client_sessions[client_id], user_input)
        else:
            # Non-streaming mode: get the full response at once
            ai_response = ask_LLM(
                model=llm_config['model'],
                system_prompt=system_prompt,
                conversation_history=conversation_history_str,
                temperature=llm_config['temperature'],
                top_p=llm_config['top_p'],
                max_tokens=llm_config['max_tokens'],
                frequency_penalty=llm_config['frequency_penalty'],
                presence_penalty=llm_config['presence_penalty'],
                streaming=False
            )
            # Split the response into sentences
            sentences = re.split(r'(?<=[.!?])\s+', ai_response)
            first_sentence = sentences[0]
            rest_of_text = ' '.join(sentences[1:])

            # Process LM activated skills on the first sentence if applicable
            if "</" in first_sentence and ">" in first_sentence:
                first_sentence, client_sessions[client_id] = process_lm_activated_skills(first_sentence, client_sessions[client_id], user_input)

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
                first_sentence, client_sessions[client_id] = process_lm_activated_skills(first_sentence, client_sessions[client_id], user_input)
            sentences = [first_sentence]
            rest_of_text = ""
            # Start TTS generation for the first sentence if not already started
            if not tts_task:
                tts_config = client_sessions[client_id].get('TTS_Config', {})
                voice = tts_config.get('voice', 'Stefanie')
                speed = tts_config.get('speed', 'normal')
                tts_task = asyncio.create_task(generate_tts(first_sentence, voice, speed, client_sessions[client_id]))

        end_time = time.time()
        print(f"LLM Time: {end_time - start_time} seconds")
        print(f"AI: {ai_response}")

        # Split the rest of the text into sentences and add to the sentences list
        if rest_of_text:
            sentences.extend(sent_tokenize(rest_of_text.strip()))

        # Join sentences to form the AI response
        ai_response = " ".join(sentences)
        # Append the AI response to the conversation history
        conversation_history.append({"role": "assistant", "content": ai_response})

        # Prepare the response data
        response_data = {
            "sentences": sentences,
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
            # Instead of raising an exception, return an error message in the response
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

        print("TIME FULL ENDPOINT SCRIPT:", time.time() - start_endpoint)
        # Print the beginning of combined data for debugging (if necessary)
        # print("combined_data:", str(combined_data)[:20000])
        # Clean up the temporary audio file
        os.remove(tts_filename)

        # Return the combined data as a response
        return Response(content=combined_data, media_type="application/octet-stream")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(content=json.dumps({"error": str(e)}).encode('utf-8'), 
                        status_code=500, 
                        media_type="application/json")

    finally:
        # Clean up the temporary audio file
        if os.path.exists(filename):
            os.remove(filename)

# Async function to generate text-to-speech audio for a given sentence
async def generate_tts(sentence: str, voice: str, speed: str, client_session: dict) -> Optional[str]:
    """
    Generate TTS audio for the given sentence.

    Parameters:
    - sentence: Text to convert to speech
    - voice: Voice to use for TTS
    - speed: Speech speed
    - client_session: Client session data

    Returns:
    - Filename of the generated audio file, or None if generation failed
    """
    logging.info(f"Generating TTS with voice: {voice}, speed: {speed} sentence: {sentence}")
    
    tts_config = client_session.get('TTS-Config', {})
    base_url = tts_config.get('TTS_SERVER_URL', 'http://213.173.96.19')
    # Call the text_to_speech function with provided parameters
    tts_output = text_to_speech(sentence, voice, speed, base_url)
    if tts_output is None:
        logging.error(f"TTS generation failed for sentence: {sentence}")
        return None
    
    # Save the TTS output to a file
    tts_filename = f"tts_output_{hash(sentence)}.wav"
    with open(tts_filename, 'wb') as tts_file:
        tts_file.write(tts_output)
    return tts_filename

# Pydantic model for the request to the /generate_tts endpoint
class SentenceRequest(BaseModel):
    sentence: str  # Sentence to convert to audio
    client_id: str  # Client ID
    voice: Optional[str] = "Stefanie"  # Voice to use (default: "Stefanie")
    speed: Optional[str] = "normal"  # Speech speed (default: "normal")

# Endpoint to generate TTS audio for a given sentence
@app.post("/generate_tts")
async def generate_tts_endpoint(request: SentenceRequest, api_key: str = Depends(verify_api_key)):
    """
    Generate text-to-speech audio for a given sentence.

    Parameters:
    - request: SentenceRequest object containing sentence, client_id, and optional voice and speed
    - api_key: API key for authentication (provided by dependency)

    Returns:
    - Audio file response
    """
    client_id = request.client_id
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    client_session = client_sessions[client_id]
    
    logger.info(f"Received TTS request for sentence: {request.sentence}")
    logger.info(f"Voice: {request.voice}, Speed: {request.speed}")
    
    # Generate TTS audio
    tts_filename = await generate_tts(request.sentence, request.voice, request.speed, client_session)

    if tts_filename is None:
        logger.error("TTS generation failed")
        raise HTTPException(status_code=500, detail="TTS generation failed")

    # Return the audio file as a response
    return FileResponse(tts_filename, media_type="audio/wav", filename=tts_filename)

# Pydantic model for the request to the /delete_tts_file endpoint
class DeleteFileRequest(BaseModel):
    filename: str  # Filename to delete

# Endpoint to delete a TTS audio file
@app.post("/delete_tts_file")
async def delete_tts_file(request: DeleteFileRequest):
    """
    Delete a TTS audio file.

    Parameters:
    - request: DeleteFileRequest object containing the filename to delete

    Returns:
    - JSONResponse indicating success or failure
    """
    try:
        if os.path.exists(request.filename):
            os.remove(request.filename)
            return JSONResponse(content={"message": "File deleted successfully"}, status_code=200)
        else:
            return JSONResponse(content={"message": "File not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"message": f"Error deleting file: {str(e)}"}, status_code=500)

# Endpoint to update the client configuration
@app.post("/update_client_config/{client_id}")
async def update_client_config(client_id: str, config: dict, api_key: str = Depends(verify_api_key)):
    """
    Update the client configuration.

    Parameters:
    - client_id: ID of the client
    - config: Configuration data to update
    - api_key: API key for authentication (provided by dependency)

    Returns:
    - JSONResponse indicating success
    """
    if client_id not in client_sessions:
        raise HTTPException(status_code=403, detail="Invalid client ID")

    # Update the client session with the new configuration
    client_sessions[client_id].update(config)
    # print(config)  # Uncomment if needed for debugging

    return JSONResponse(content={"message": "Client configuration updated successfully"}, status_code=200)

# Endpoint to get client data
@app.get("/get_client_data/{client_id}")
async def get_client_data_endpoint(client_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get the client data for a given client ID.

    Parameters:
    - client_id: ID of the client
    - api_key: API key for authentication (provided by dependency)

    Returns:
    - JSONResponse containing the client data
    """
    if client_id not in client_sessions:
        raise HTTPException(status_code=404, detail="Client not found")
    return JSONResponse(content=client_sessions[client_id], status_code=200)

# -----------------------------------------------------------------------------
# Server Run Logic
# -----------------------------------------------------------------------------

# Function to print the server menu
def print_menu():
    """Print the server menu options."""
    print("1. Exit")
    print("Enter your choice (1): ", end="", flush=True)

# Function to run the command-line interface
def run_cli():
    """Run the command-line interface for the server."""
    global is_running
    while is_running:
        print_menu()
        choice = input()
        if choice == "1":
            is_running = False
            break
        else:
            print("Invalid choice. Please try again.")

# Function to run the FastAPI server
def run_server():
    """Run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8001)

# Function to clean up old TTS files periodically
def cleanup_old_tts_files():
    """
    Periodically clean up old TTS audio files that are older than a specified time.
    """
    while True:
        current_time = time.time()
        for filename in os.listdir('.'):
            if filename.startswith('tts_output') and filename.endswith('.wav'):
                file_path = os.path.join('.', filename)
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 60:  # Delete files older than 60 seconds
                    try:
                        os.remove(file_path)
                        print(f"Deleted old file: {filename}")
                    except Exception as e:
                        print(f"Error deleting {filename}: {e}")
        time.sleep(60)  # Sleep for 60 seconds before the next cleanup

# Main entry point for running the server and CLI
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