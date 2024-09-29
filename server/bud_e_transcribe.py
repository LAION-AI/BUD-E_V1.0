import requests
import time

"""
Version 1: FastAPI Server with Hugging Face Whisper Model

This version sends a WAV file to a FastAPI server that uses a Hugging Face 
Whisper model for automatic speech recognition (ASR). It's designed for 
direct integration with a custom ASR server.
"""

def transcribe(file_path, client_id="123", server_url="http://213.173.96.19:8011/transcribe/"):
    """
    Send a WAV file to the transcription server and get the transcription back.
    
    :param file_path: Path to the WAV file
    :param client_id: Identifier for the client (not used in this version)
    :param server_url: URL of the transcription server
    :return: Transcription text
    """
    try:
        # Start timing the entire process
        start_time = time.time()
        
        # Open the audio file in binary mode
        with open(file_path, 'rb') as audio_file:
            # Create a dictionary with the file for the POST request
            files = {'file': (file_path, audio_file, 'audio/wav')}
            
            # Send the POST request to the server
            response = requests.post(server_url, files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Calculate and print timing information
                end_time = time.time()
                total_time = end_time - start_time
                print(f"\nServer-side transcription time: {result['transcription_time']}")
                print(f"Total request time: {total_time:.2f} seconds")            
                
                # Return only the transcription text
                return result['transcription']
            else:
                # Handle error cases
                print(f"Error: Server returned status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
    except Exception as e:
        # Handle any exceptions that occur during the process
        print(f"An error occurred: {str(e)}")
        return None

'''
Version 2: Faster Whisper Server

This version uses a custom OpenAI-compatible API that serves a faster 
version of Whisper. It's designed for improved transcription speed while 
maintaining compatibility with OpenAI's API structure.

from openai import OpenAI
import time

# Initialize OpenAI client with custom base URL
client = OpenAI(api_key="cant-be-empty", base_url="http://213.173.96.19:8005/v1/")

def transcribe(filename, client_id):
    """
    Transcribe an audio file using a faster Whisper model.
    
    :param filename: Path to the audio file
    :param client_id: Identifier for the client, used in logging
    :return: Transcription text
    """
    try:
        # Open and read the audio file
        with open(filename, "rb") as file:
            # Send transcription request to the server
            transcription = client.audio.transcriptions.create(
                model="ddorian43/whisper-tiny-int8",  # Specific model for faster transcription
                file=(filename, file.read())
            )
            # Log the transcription result with timestamp and client ID
            print(f"Client {client_id} - {time.strftime('%Y-%m-%d %H:%M:%S')}: {transcription.text}")
            return transcription.text
    except Exception as e:
        # Handle and log any errors
        print(f"Client {client_id} - Error: {e}")
'''

'''
Version 3: Groq API Integration

This version demonstrates how to use the Groq API for transcription. 
Groq is known for its high-performance AI infrastructure, potentially 
offering faster processing times for large language models and related tasks.

import time
from groq import Groq
import os

# Set up Groq API key and initialize client
os.environ['GROQ_API_KEY'] = 'xxxx'  
client = Groq()

def transcribe(filename, client_id):
    """
    Transcribe an audio file using the Groq API.
    
    :param filename: Path to the audio file
    :param client_id: Identifier for the client, used in logging
    :return: Transcription text
    """
    try:
        # Open and read the audio file
        with open(filename, "rb") as file:
            # Send transcription request to Groq API
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                response_format="json",
                # Optional parameters:
                # prompt="Specify context or spelling",
                # language="en",
                # temperature=0.0
            )
            # Log the transcription result with timestamp and client ID
            print(f"Client {client_id} - {time.strftime('%Y-%m-%d %H:%M:%S')}: {transcription.text}")
            return transcription.text
    except Exception as e:
        # Handle and log any errors
        print(f"Client {client_id} - Error: {e}")
'''