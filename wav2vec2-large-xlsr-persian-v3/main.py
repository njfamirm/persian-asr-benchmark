import sys
import json
import torch
from io import StringIO
from datasets import load_dataset
from metrics import calculate_wer
from normalizer import normalizer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf  # Add this import to read audio files

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Load the audio file provided as a command-line argument
audio_file_path = sys.argv[1]
audio_input, sample_rate = sf.read(audio_file_path)

# Process the audio input
result = pipe(audio_input)
transcription = result["text"]
print(transcription)

# Save the transcription result to a JSON file
output = {
    "transcription": transcription,
    # Placeholder values for WER calculations
    "werBeforeNormalization": None,
    "normalizedTranscription": None,
    "werAfterNormalization": None
}
with open("result.json", "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)