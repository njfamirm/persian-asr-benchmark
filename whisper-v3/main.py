import os
import torch
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pydub import AudioSegment
from io import BytesIO

# Paths
original_audio_path = '../assets/audio-01.wav'
trimmed_audio_path = 'trimmed_audio.wav'

# Check if the audio file exists
if not os.path.exists(original_audio_path):
    raise FileNotFoundError(f"Audio file not found: {original_audio_path}")

# Load and trim the audio file
try:
    audio = AudioSegment.from_wav(original_audio_path)
    audio_29s = audio[:29 * 1000]  # pydub works in milliseconds
    with open(trimmed_audio_path, "wb") as out_file:
        audio_29s.export(out_file, format="wav")
except Exception as e:
    raise RuntimeError(f"Audio processing failed: {e}")

# Check if the trimmed audio file was created
if not os.path.exists(trimmed_audio_path):
    raise FileNotFoundError(f"Trimmed audio file not created: {trimmed_audio_path}")

# Setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
model_id = "openai/whisper-large-v3-turbo"
try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Initialize the pipeline
try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
except Exception as e:
    raise RuntimeError(f"Pipeline initialization failed: {e}")

# Use the pipeline on the trimmed audio
try:
    result = pipe(trimmed_audio_path)
    transcription_text = result["text"]
    print(transcription_text)
except Exception as e:
    raise RuntimeError(f"Pipeline inference failed: {e}")

# Prepare the output data
output_data = {
    "text": transcription_text
}

# Write JSON output to a file
json_file_path = 'result.json'
try:
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
except Exception as e:
    raise RuntimeError(f"Failed to write JSON file: {e}")

# Verify if the JSON file was created
if not os.path.exists(json_file_path):
    raise FileNotFoundError(f"JSON file not created: {json_file_path}")

# Read and print the JSON output
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_output = f.read()
    print(json_output)
except Exception as e:
    raise RuntimeError(f"Failed to read JSON file: {e}")