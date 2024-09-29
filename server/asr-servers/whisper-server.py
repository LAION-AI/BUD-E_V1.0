from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import time
import io
import soundfile as sf
import librosa

app = FastAPI()

# Global variables for the model and pipeline
model = None
processor = None
pipe = None

def load_model(model_name= "openai/whisper-medium"):  #"aware-ai/whisper-base-german"):
    global model, processor, pipe
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    if device == "cuda:0":
        try:
            model = model.half().to(device)  # FP16
            print("Using FP16 quantization")
        except RuntimeError:
            try:
                model = model.to(device).to(torch.int8)  # INT8
                print("Using INT8 quantization")
            except RuntimeError:
                print("Quantization not supported, using full precision")
                model = model.to(device)
    else:
        model = model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if file.filename.endswith('.wav') or file.filename.endswith('.mp3'):
        try:
            # Read the file content
            content = await file.read()
            audio_data, sample_rate = sf.read(io.BytesIO(content))
            
            # If MP3, convert to WAV
            if file.filename.endswith('.mp3'):
                audio_data, sample_rate = librosa.load(io.BytesIO(content), sr=None)
            
            # Measure transcription time
            start_time = time.time()
            
            # Transcribe the audio
            result = pipe({"array": audio_data, "sampling_rate": sample_rate})
            
            # Calculate transcription duration
            end_time = time.time()
            transcription_time = end_time - start_time
            
            return JSONResponse(content={
                "transcription": result["text"],
                "transcription_time": f"{transcription_time:.2f} seconds"
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a WAV or MP3 file.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
