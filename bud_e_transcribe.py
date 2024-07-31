
from openai import OpenAI
import time
client = OpenAI(api_key="cant-be-empty", base_url="http://213.173.96.19:8005/v1/")

def transcribe(filename, client_id):
    try:
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
            model="ddorian43/whisper-tiny-int8", file=(filename, file.read())
            )
            print(f"Client {client_id} - {time.strftime('%Y-%m-%d %H:%M:%S')}: {transcription.text}")
            return transcription.text
    except Exception as e:
        print(f"Client {client_id} - Error: {e}")


''' 
import time
from groq import Groq
import os

# Initialize Groq client
os.environ['GROQ_API_KEY'] = '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'  # Replace 'xx' with your actual API key
client = Groq()

def transcribe(filename, client_id):
    try:
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                #prompt="Specify context or spelling",  # Optional
                response_format="json",  # Optional
                #language="en",  # Optional
                #temperature=0.0  # Optional
            )
            print(f"Client {client_id} - {time.strftime('%Y-%m-%d %H:%M:%S')}: {transcription.text}")
            return transcription.text
    except Exception as e:
        print(f"Client {client_id} - Error: {e}")


''' 
