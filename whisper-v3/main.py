import os
import torch
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pydub import AudioSegment

# Paths
original_audio_path = '../assets/audio-01.wav'
trimmed_audio_path = 'trimmed_audio.wav'

# Check if the audio file exists
if not os.path.exists(original_audio_path):
    raise FileNotFoundError(f"Audio file not found: {original_audio_path}")

# Load and trim the audio file
audio = AudioSegment.from_wav(original_audio_path)
audio_29s = audio[:29 * 1000]  # pydub works in milliseconds

# Export the trimmed audio to a temporary file
audio_29s.export(trimmed_audio_path, format="wav")

# Check if the trimmed audio file was created
if not os.path.exists(trimmed_audio_path):
    raise FileNotFoundError(f"Trimmed audio file not created: {trimmed_audio_path}")

# Setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the model and processor
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Initialize the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Use the pipeline on the trimmed audio
try:
    result = pipe(trimmed_audio_path)
    print(result["text"])
except Exception as e:
    raise RuntimeError(f"Pipeline failed with error: {e}")

# Prepare the output data
output_data = {
    "text": result["text"]
}

# Write JSON output to a file
json_file_path = 'result.json'
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# Verify if the JSON file was created
if not os.path.exists(json_file_path):
    raise FileNotFoundError(f"JSON file not created: {json_file_path}")

# Read and print the JSON output
with open(json_file_path, 'r', encoding='utf-8') as f:
    json_output = f.read()

print(json_output)