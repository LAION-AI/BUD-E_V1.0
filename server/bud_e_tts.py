"""
DeepGram Text-to-Speech Module for BUD-E

This module provides a text-to-speech (TTS) interface using the DeepGram API.
It offers a simple way to convert text into speech, with options for different
voices and models.

Key features:
- Integration with DeepGram's TTS API
- Support for multiple voices and models
- Customizable speech parameters
- Error handling and logging

Note: This implementation assumes you have set up a DeepGram account and have
an API key. The API key should be stored in an environment variable or a secure
configuration file.
"""

import os
from dotenv import load_dotenv
#from deepgram import DeepgramClient, SpeakOptions
import time
import io
import requests

# Load environment variables (including the API key)
load_dotenv()

# DeepGram API key (replace with your actual key or use an environment variable)
DEEPGRAM_API_KEY = "xxx"

def text_to_speech(text, voice_name, speed, base_url):
    """
    Convert text to speech using the DeepGram Text-to-Speech API.
    
    Args:
    text (str): The text to be converted to speech
    voice_name (str): Name of the voice model to use (default: "aura-luna-en")
    
    Returns:
    bytes: Audio data of the generated speech, or None if an error occurs
    """
    # Set default voice for "stephanie"
    if voice_name.lower() == "stefanie":
        voice_name = "aura-luna-en"
    if voice_name.lower() == "florian":
        voice_name = "aura-helios-en"


    try:
        # Start timing the execution
        start_time = time.time()

        # Prepare headers and data for the request
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "text": text,
        }

        # Deepgram TTS endpoint with the specified voice model
        url = f"https://api.deepgram.com/v1/speak?model={voice_name}"

        # Make the request to Deepgram TTS API
        response = requests.post(url, headers=headers, json=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Calculate and print the latency
            latency = time.time() - start_time
            print(f"Latency: {latency:.2f} seconds")

            # Return the audio content (binary data)
            return response.content
        else:
            print(f"Error in text_to_speech: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return None









'''

"""
This module provides a text-to-speech (TTS) interface for the Fish TTS system.
It handles the process of converting text input to speech output, including:
- Language detection of the input text
- Selection of appropriate reference audio based on the detected language
- Interfacing with the Fish TTS API
- Post-processing of the generated audio (trimming, fading)

The main function, text_to_speech, coordinates these steps and returns the
processed audio data. This module ensures that the correct reference audio
and text are submitted to the Fish TTS endpoint, which is crucial for
maintaining the quality and consistency of the generated speech across
different languages.

Key features:
- Multi-language support with automatic language detection
- Dynamic reference audio selection based on detected language
- Audio post-processing for improved output quality
- Error handling and fallback mechanisms
"""

# Import necessary libraries
import requests  # For making HTTP requests to the TTS API
import io  # For handling byte streams
import time  # For measuring execution time
import ormsgpack  # For efficient serialization of request data
from pathlib import Path  # For file path operations
from pydub import AudioSegment  # For audio processing
from fast_langdetect import detect  # For language detection

# Dictionary mapping language codes to their corresponding reference audio files
# These reference files are crucial for the Fish TTS system to generate speech in the correct language and style
LANGUAGE_REFERENCES = {
    "en": "juniper-long-en.wav",
    "de": "juniper-long-de.wav",
    "ar": "juniper-long-ar.wav",
    "ko": "juniper-long-ko.wav",
    "ja": "juniper-long-jp.wav",
    "zh": "juniper-long-zh.wav",
    "es": "juniper-long-es.wav",
    "fr": "juniper-long-fr.wav"
}

def read_file_with_fallback_encoding(file_path):
    """
    Attempt to read a file with different encodings to handle potential encoding issues.
    This is particularly useful for reading reference text files which may have varying encodings.
    
    Args:
    file_path (str): Path to the file to be read

    Returns:
    str: Content of the file

    Raises:
    ValueError: If the file cannot be read with any of the attempted encodings
    """
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read file {file_path} with any of the attempted encodings")

def text_to_speech(text, voice_name, speed, base_url="http://213.173.96.19"):
    """
    Main function to convert text to speech using the Fish TTS system.
    
    Args:
    text (str): The text to be converted to speech
    voice_name (str): Name of the voice to be used (currently not utilized in the function)
    speed (str): Speed of the speech (currently not utilized in the function)
    base_url (str): Base URL of the TTS API

    Returns:
    bytes: Audio data of the generated speech, or None if an error occurs
    """
    try:
        # Clean up input text by removing newlines
        text = text.replace('\n', '').replace('\r', '')

        # Start timing the execution
        s = time.time()
        print("######################################")
        print("voice_name", voice_name)

        # Normalize voice name and speed (not currently used in the function)
        voice_name = voice_name.lower().strip()
        speed = speed.lower().strip()

        # Detect the language of the input text
        detected_lang = detect(text)["lang"]
        print(f'Detected language: {detected_lang}')

        # Check if the detected language is supported, default to English if not
        if detected_lang not in LANGUAGE_REFERENCES:
            print(f"Unsupported language: {detected_lang}. Defaulting to English.")
            detected_lang = "en"

        # Get the corresponding reference audio file path
        reference_audio = f"./{LANGUAGE_REFERENCES[detected_lang]}"
        reference_text_file = reference_audio.replace('.wav', '.txt')

        # Read the reference text from the corresponding .txt file
        try:
            reference_text = read_file_with_fallback_encoding(reference_text_file)
        except ValueError as e:
            print(f"Error reading reference text: {e}")
            print("Using a default reference text.")
            reference_text = "This is a default reference text."

        # Generate speech using the texttospeech function
        output_path = texttospeech(
            text,
            url=f"{base_url}:8080/v1/tts",
            reference_audio=reference_audio,
            reference_text=reference_text,
            output="generated",
            format="wav",
            streaming=False,
            normalize=True,
            mp3_bitrate=64,
            opus_bitrate=-1000,
            max_new_tokens=1024,
            chunk_length=100,
            top_p=0.9,
            repetition_penalty=1.25,
            temperature=0.65,
            speaker=None,
            emotion=None
        )

        if output_path is None:
            raise ValueError("Speech generation failed")

        # Process the audio file
        audio = AudioSegment.from_wav(output_path)
        
        # Trim the first and last 50ms to remove potential artifacts
        audio = audio[50:-50]
        
        # Apply fade in and fade out for smoother start and end
        audio = audio.fade_in(duration=80).fade_out(duration=80)
        
        # Export the processed audio
        processed_output_path = output_path.replace('.wav', '_processed.wav')
        audio.export(processed_output_path, format="wav")
        
        # Read and return the processed audio file content
        with open(processed_output_path, "rb") as audio_file:
            audio_data = audio_file.read()
        print(f"Latency: {time.time() - s:.2f} seconds")
        return audio_data

    except Exception as e:
        print(f"Error in text_to_speech: {e}")
    return None

# The following classes and functions are used to interface with the Fish TTS API

class ServeReferenceAudio:
    """
    Class to represent reference audio data for the TTS API request.
    """
    def __init__(self, audio, text):
        self.audio = audio
        self.text = text

    def to_dict(self):
        return {"audio": self.audio, "text": self.text}

class ServeTTSRequest:
    """
    Class to represent a TTS API request, with methods to convert to a dictionary.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, ServeReferenceAudio) else v for k, v in self.__dict__.items()}

def audio_to_bytes(file_path):
    """
    Read an audio file and return its content as bytes.
    
    Args:
    file_path (str): Path to the audio file

    Returns:
    bytes: Content of the audio file
    """
    with open(file_path, "rb") as audio_file:
        return audio_file.read()

def texttospeech(
    text,
    url="http://213.173.96.19:8080/v1/tts",
    reference_audio="/mnt/raid/spirit/fish-speech/49.wav",
    reference_text="Der Saturnring erstreckt sich ueber eine Distanz von 282.000 Kilometern.",
    output="generated",
    format="wav",
    streaming=False,
    normalize=True,
    mp3_bitrate=64,
    opus_bitrate=-1000,
    max_new_tokens=1024,
    chunk_length=100,
    top_p=0.9,
    repetition_penalty=1.3,
    temperature=0.65,
    speaker=None,
    emotion=None
):
    """
    Generate speech from text using the specified TTS service.
    
    This function prepares the request data and sends it to the Fish TTS API.
    It handles the API response and saves the generated audio to a file.

    Args:
    text (str): The text to be converted to speech
    url (str): The URL of the TTS API
    reference_audio (str): Path to the reference audio file
    reference_text (str): The text corresponding to the reference audio
    output (str): Base name for the output audio file
    format (str): Output audio format (e.g., 'wav')
    ... (other parameters control various aspects of the TTS generation)

    Returns:
    str: Path to the generated audio file, or None if the request failed
    """
    # Read the reference audio file
    ref_audio = audio_to_bytes(reference_audio)

    # Prepare the request data
    data = {
        "text": text,
        "references": [ServeReferenceAudio(audio=ref_audio, text=reference_text).to_dict()],
        "reference_id": None,
        "normalize": normalize,
        "format": format,
        "mp3_bitrate": mp3_bitrate,
        "opus_bitrate": opus_bitrate,
        "max_new_tokens": max_new_tokens,
        "chunk_length": chunk_length,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "speaker": speaker,
        "emotion": emotion,
        "streaming": streaming,
    }

    # Create a ServeTTSRequest object and convert it to a dictionary
    pydantic_data = ServeTTSRequest(**data)

    # Send the request to the TTS API
    response = requests.post(
        url,
        data=ormsgpack.packb(pydantic_data.to_dict()),
        stream=streaming,
        headers={
            "authorization": "Bearer YOUR_API_KEY",
            "content-type": "application/msgpack",
        },
    )

    # Handle the API response
    if response.status_code == 200:
        audio_content = response.content
        audio_path = f"{output}.{format}"
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_content)
        print(f"Audio has been saved to '{audio_path}'.")
        return audio_path
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.json())
        return None






"""
This module provides a text-to-speech (TTS) interface that connects to various TTS services.
It supports multiple voices and TTS engines:

1. Stefanie and Florian: Azure TTS from Microsoft
2. Thorsten: German DDC TTS running on a server
3. Stella: StyleTTS2 server

The main function, text_to_speech, selects the appropriate TTS service based on the
requested voice and sends the text to be synthesized. It then retrieves and returns
the generated audio data.

Note: This code assumes that the necessary TTS servers are set up and running.
Example code for setting up these servers can be found in the BUD-E repository.
"""

import requests  # For making HTTP requests to the TTS servers
import io  # For handling byte streams (not used in this version)
import sounddevice as sd  # For audio playback (not used in this version)
import soundfile as sf  # For audio file handling (not used in this version)
import time  # For measuring execution time

def text_to_speech(text, voice_name, speed, base_url="http://213.173.96.19"):
    """
    Convert text to speech using various TTS services based on the selected voice.

    Args:
    text (str): The text to be converted to speech
    voice_name (str): Name of the voice to be used (e.g., "stefanie", "florian", "thorsten", "stella")
    speed (str): Speed of the speech (only applicable for some voices)
    base_url (str): Base URL of the TTS servers

    Returns:
    bytes: Audio data of the generated speech, or None if an error occurs
    """
    try:
        # Start timing the execution
        s = time.time()
        print("######################################")
        print("voice_name", voice_name)

        # Normalize voice name and speed to ensure consistent formatting
        voice_name = voice_name.lower().strip()
        speed = speed.lower().strip()
        
        # Determine the correct endpoint and parameters based on the voice
        if voice_name.lower() in ["stefanie", "florian"]:
            # Azure TTS from Microsoft
            url = f"{base_url}:5004/synthesize"
            params = {"text": text, "voice": voice_name.capitalize(), "speed": speed}
        elif voice_name == "thorsten":
            # German DDC TTS server
            url = f"{base_url}:5006/synthesize"
            params = {"text": text}
        elif voice_name == "stella":
            # StyleTTS2 server
            url = f"{base_url}:5001/synthesize"
            params = {"text": text}
        else:
            # Default to Stefanie if an invalid voice is specified
            print(f"Invalid voice name: {voice_name}. Defaulting to Stefanie.")
            url = f"{base_url}:5004/synthesize"
            params = {"text": text, "voice": "Stefanie", "speed": "normal"}

        print(f"Sending request to {url} with params: {params}")
        
        # Send the request to the appropriate TTS server
        response = requests.post(url, json=params)
        
        if response.status_code == 200:
            # Extract the filename of the generated audio from the response
            filename = response.json()['filename']
            print(f"Audio file generated: {filename}")
            
            # Construct the URL for downloading the audio file
            audio_url = f"{url.rsplit('/', 1)[0]}/audio/{filename}"
            
            # Download the audio file
            audio_response = requests.get(audio_url)
            
            if audio_response.status_code == 200:
                # Load the audio data
                audio_data = audio_response.content
                print(f"Latency: {time.time() - s:.2f} seconds")
                return audio_data
            else:
                print(f"Failed to download audio file. Status code: {audio_response.status_code}")
        else:
            print(f"Failed to synthesize speech. Status code: {response.status_code}")
            print(response.text)  # Use response.text for more general error messages
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
    return None
'''
