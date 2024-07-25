from fastapi import FastAPI, File, UploadFile

import torch
import numpy as np
import io
from pydub import AudioSegment
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Silero VAD model
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()
    return sound

def process_audio(audio_data):
    logger.info(f"Processing audio: shape={audio_data.shape}")

    # Ensure the audio is mono
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16000 Hz if necessary
    if audio_data.shape[0] % 16000 != 0:
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=44100,
            sample_width=audio_data.dtype.itemsize,
            channels=1
        )
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_data = np.array(audio_segment.get_array_of_samples())

    # Convert to float
    audio_float = int2float(audio_data)

    logger.info(f"Processed audio: shape={audio_float.shape}")
    return audio_float

@app.post("/vad")
async def vad(data: UploadFile = File(...)):
    try:
        audio_data = np.frombuffer(await data.read(), dtype=np.int16)
        audio_float = process_audio(audio_data)

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(audio_float)

        logger.info(f"Tensor created: shape={tensor.shape}")

        # Prepare input for the model
        num_samples = 512  # Silero VAD expects 512 samples for 16000 Hz
        confidence_scores = []

        # Process audio in chunks
        for i in range(0, len(tensor), num_samples):
            chunk = tensor[i:i+num_samples]
            if len(chunk) < num_samples:
                chunk = torch.nn.functional.pad(chunk, (0, num_samples - len(chunk)))
            confidence = model(chunk, 16000).item()
            confidence_scores.append(confidence)

        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        logger.info(f"VAD processing complete: avg_confidence={avg_confidence}")

        return {"confidence": avg_confidence}

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise Exception(f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
