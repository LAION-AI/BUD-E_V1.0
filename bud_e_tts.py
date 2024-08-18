import requests
import io
import sounddevice as sd
import soundfile as sf
import time

def text_to_speech(text, voice_name, speed, base_url="http://213.173.96.19"):
    try:
        s = time.time()
        print("######################################")
        print("voice_name", voice_name)
        # Normalize voice name and speed
        voice_name = voice_name.lower().strip()
        speed = speed.lower().strip()
        
        # Determine the correct endpoint and parameters based on the voice
        if voice_name.lower() in ["stefanie", "florian"]:
            url = f"{base_url}:5004/synthesize"
            params = {"text": text, "voice": voice_name.capitalize(), "speed": speed}
        elif voice_name == "thorsten":
            url = f"{base_url}:5006/synthesize"
            params = {"text": text}
        elif voice_name == "stella":
            url = f"{base_url}:5001/synthesize"
            params = {"text": text}
        else:
            print(f"Invalid voice name: {voice_name}. Defaulting to Stefanie.")
            url = f"{base_url}:5004/synthesize"
            params = {"text": text, "voice": "Stefanie", "speed": "normal"}

        print(f"Sending request to {url} with params: {params}")

        # Send the request to the server
        response = requests.post(url, json=params)
        
        if response.status_code == 200:
            filename = response.json()['filename']
            print(f"Audio file generated: {filename}")
            
            # Download the audio file
            audio_url = f"{url.rsplit('/', 1)[0]}/audio/{filename}"
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
            print(response.text)  # Use response.text instead of response.json() for more general error messages
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
    return None









'''


import requests
import io
import sounddevice as sd
import soundfile as sf
import time


THORSTEN_DDC_SERVER_URL = "http://213.173.96.19:5004"  # Replace with your server's IP and port

def text_to_speech(text):
    try:
        s = time.time()
        # Send the text to the server
        response = requests.post(f"{THORSTEN_DDC_SERVER_URL}/synthesize", json={"text": text})

        if response.status_code == 200:
            filename = response.json()['filename']
            print(f"Audio file generated: {filename}")

            # Download the audio file
            audio_response = requests.get(f"{THORSTEN_DDC_SERVER_URL}/audio/{filename}")

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






import requests
import io
import sounddevice as sd
import soundfile as sf
import time

PIPER_SERVER_URL = "http://213.173.96.19:5003"  # Update this if your server port is different

def text_to_speech(text):
    try:
        s = time.time()
        # Send the text to the server
        response = requests.post(f"{PIPER_SERVER_URL}/tts", json={"text": text})
        
        if response.status_code == 200:
            # Load the audio data
            audio_data = response.content
            
            # Get audio parameters from headers
            sample_rate = int(response.headers.get('X-Sample-Rate', 22000))
            channels = int(response.headers.get('X-Channels', 1))
            
            print("Latency:", time.time() - s)
            return audio_data
        else:
            print(f"Failed to synthesize speech. Status code: {response.status_code}")
            print(response.text)  # Print the error message from the server
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
    return None
'''
