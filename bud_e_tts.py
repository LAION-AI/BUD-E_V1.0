
import requests
import io
import sounddevice as sd
import soundfile as sf
import time


STYLE_TTS_SERVER_URL = "http://213.173.96.19:5001"  # Replace with your server's IP and port

def text_to_speech(text):
    try:
        s = time.time()
        # Send the text to the server
        response = requests.post(f"{STYLE_TTS_SERVER_URL}/synthesize", json={"text": text})

        if response.status_code == 200:
            filename = response.json()['filename']
            print(f"Audio file generated: {filename}")

            # Download the audio file
            audio_response = requests.get(f"{STYLE_TTS_SERVER_URL}/audio/{filename}")

            if audio_response.status_code == 200:
                # Load the audio data
                audio_data = audio_response.content
                print("Latency:", time.time() - s)
                return audio_data
            else:
                print(f"Failed to download audio file. Status code: {audio_response.status_code}")
        else:
            print(f"Failed to synthesize speech. Status code: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
    return None

'''

import io
from dimits import Dimits
import time
import soundfile as sf

# Initialize Dimits with the desired voice model
dt = Dimits("en_US-amy-low")

def text_to_speech(text):
    """Function to convert text to speech using Dimits TTS."""
    try:
        start_time = time.time()
        
        # Generate a unique filename for this audio
        filename = f"tts_output_{int(time.time())}.wav"
        file_path = f"./{filename}"
        
        # Generate audio file
        dt.text_2_audio_file(text, filename, "./", format="wav")
        
        # Read the generated audio file and convert to the format expected by the server
        with sf.SoundFile(file_path) as sound_file:
            audio_data = sound_file.read()
            sample_rate = sound_file.samplerate
        
        # Create an in-memory bytes buffer
        audio_buffer = io.BytesIO()
        
        # Write the audio data to the buffer in WAV format
        sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
        
        # Get the content of the buffer
        audio_content = audio_buffer.getvalue()
        
        end_time = time.time()
        print(f"Time taken for TTS generation: {end_time - start_time:.4f} seconds")
        
        return audio_content
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return None

'''