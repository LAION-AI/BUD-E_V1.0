import argparse
import base64
import wave
import time
import statistics
from pathlib import Path

import ormsgpack
import requests
from pydub import AudioSegment
from pydub.playback import play

class ServeReferenceAudio:
    def __init__(self, audio, text):
        self.audio = audio
        self.text = text

    def to_dict(self):
        return {"audio": self.audio, "text": self.text}

class ServeTTSRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, ServeReferenceAudio) else v for k, v in self.__dict__.items()}

def audio_to_bytes(file_path):
    with open(file_path, "rb") as audio_file:
        return audio_file.read()

def read_ref_text(file_path):
    with open(file_path, "r") as text_file:
        return text_file.read().strip()

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
    top_p=0.8,
    repetition_penalty=1.2,
    temperature=0.75,
    speaker=None,
    emotion=None
):
    ref_audio = audio_to_bytes(reference_audio)
    ref_text = read_ref_text(reference_audio.replace('.wav', '.txt'))

    data = {
        "text": text,
        "references": [ServeReferenceAudio(audio=ref_audio, text=ref_text).to_dict()],
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

    pydantic_data = ServeTTSRequest(**data)

    response = requests.post(
        url,
        data=ormsgpack.packb(pydantic_data.to_dict()),
        stream=streaming,
        headers={
            "authorization": "Bearer YOUR_API_KEY",
            "content-type": "application/msgpack",
        },
    )

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

def main():
    text = "Ich bin hier, um dir zu helfen, deine Schulziele zu erreichen und dich beim Lernen zu unterstuetzen. "
    times = []

    for i in range(10):
        start_time = time.time()
        output_path = texttospeech(text, output=f"generated_{i}")
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"Iteration {i+1}: Inference time = {inference_time:.2f} seconds")

    mean_time = statistics.mean(times)
    print(f"\nMean inference time over 10 iterations: {mean_time:.2f} seconds")

if __name__ == "__main__":
    main()