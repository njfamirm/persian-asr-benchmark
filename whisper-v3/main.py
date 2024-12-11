import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pydub import AudioSegment

# Load the audio file
audio = AudioSegment.from_wav('../assets/audio-01.wav')

# Trim the audio to the first 29 seconds
audio_29s = audio[:29 * 1000]  # pydub works in milliseconds

# Export the trimmed audio to a temporary file
audio_29s.export("trimmed_audio.wav", format="wav")

audio_path = '../assets/audio-01.wav'
ground_truth_path = '../assets/audio-01.txt'

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

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# Prepare the output data
output_data = {
    "text": result["text"]
}
print(output_data)

# Write JSON output to a file
with open('result.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# Read and print the JSON output
with open('result.json', 'r', encoding='utf-8') as f:
    json_output = f.read()
    print(json_output)