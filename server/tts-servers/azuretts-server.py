from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import azure.cognitiveservices.speech as speechsdk
import os
import random
import re

app = FastAPI()

# Azure TTS configuration
api_key = 'xxxxx'
region = 'germanywestcentral'

class TextInput(BaseModel):
    text: str
    voice: str = "Stefanie"  # Default voice
    speed: str = "normal"    # Default speed

from fuzzywuzzy import process

def get_voice_name(voice):
    """Map input voice to the correct Azure voice name, resilient to spelling variations and case-insensitive."""
    normalized_voice = normalize_input(voice)
    
    voice_mapping = {
        "stefanie": "de-DE-SeraphinaMultilingualNeural",
        "florian": "de-DE-FlorianMultilingualNeural",
        # Add more voice mappings as needed
    }
    
    # List of common variations for Stefanie and Florian
    voice_variations = {
        "stefanie": ["stefanie", "stephanie", "stephany", "stefany", "steffanie", "stephanie", 
                     "steffi", "steffy", "steffi", "stef", "steph"],
        "florian": ["florian", "florean", "florien", "floryan", "florijan", "flo", "florin"]
    }
    
    # Check for exact matches or variations first (case-insensitive)
    for key, variations in voice_variations.items():
        if normalized_voice in [normalize_input(v) for v in variations]:
            return voice_mapping[key]
    
    # Use fuzzy matching to find the closest match
    closest_match, score = process.extractOne(normalized_voice, list(voice_mapping.keys()))
    
    # If the match score is above a threshold (e.g., 80), use that voice
    if score >= 80:
        return voice_mapping[closest_match]
    
    # If no good match is found, default to Stefanie
    print(f"No close match found for '{voice}'. Defaulting to Stefanie.")
    return voice_mapping["stefanie"]

# Update the normalize_input function to be more lenient and case-insensitive
def normalize_input(input_str):
    """Normalize input string to handle typos, variations, and case differences."""
    return ''.join(char.lower() for char in input_str if char.isalnum())



def get_speed_rate(speed):
    """Map input speed to SSML rate value."""
    normalized_speed = normalize_input(speed)
    speed_mapping = {
        "langsam": "-20%",
        "normal": "0%",
        "schnell": "+20%",
        # Add more speed mappings as needed
    }
    return speed_mapping.get(normalized_speed, "0%")  # Default to normal if not found

def text_to_speech(text, voice_name, speed_rate, output_audio_path):
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_synthesis_voice_name = voice_name
    
    ssml = f"""
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="de-DE">
        <voice name="{voice_name}">
            <prosody rate="{speed_rate}">
                {text}
            </prosody>
        </voice>
    </speak>
    """
    
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_audio_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized for text [{text}] with voice {voice_name} and speed {speed_rate}")
        return True
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(f"Error details: {cancellation_details.error_details}")
        return False


@app.post("/synthesize")
async def synthesize_speech(text_input: TextInput):
    text = text_input.text
    voice = text_input.voice
    speed = text_input.speed
    print(f"Received request: text='{text}', voice='{voice}', speed='{speed}'")
    
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    
    try:
        voice_name = get_voice_name(voice)
        print(f"Input voice: '{voice}', Normalized input: '{normalize_input(voice)}', Mapped voice name: {voice_name}")
        speed_rate = get_speed_rate(speed)        
        # Generate a random filename
        random_number = random.randint(10000, 99999)
        filename = f"speech_{random_number}.wav"
        
        # Ensure the 'audio' directory exists
        os.makedirs('audio', exist_ok=True)
        
        # Full path for the output file
        file_path = os.path.join('audio', filename)
        
        # Generate speech and save to file
        success = text_to_speech(text, voice_name, speed_rate, file_path)
        
        if success:
            return JSONResponse(content={"success": True, "filename": filename}, status_code=200)
        else:
            return JSONResponse(content={"success": False, "message": "Failed to synthesize speech"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"success": False, "message": f"An error occurred: {str(e)}"}, status_code=500)

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.join('audio', filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Audio file not found")

if __name__ == '__main__':
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=5004)