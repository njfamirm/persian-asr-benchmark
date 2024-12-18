import sys
import json
from io import StringIO
from predict import load_model, predict
from metrics import calculate_wer

processor, model = load_model()

audio_path = '../assets/audio-01.wav'
ground_truth_path = '../assets/audio-01.txt'

# Read the ground truth text
with open(ground_truth_path, 'r') as file:
    ground_truth = file.read()

transcription = predict(processor, model, audio_path)

# Calculate WER before normalizing
wer_score_before = calculate_wer(ground_truth, transcription)

output_data = {
    'transcription': transcription,
    'wer': wer_score_before,
}

# Write JSON output to a file
with open('result.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# Read and print the JSON output
with open('result.json', 'r', encoding='utf-8') as f:
    json_output = f.read()

print(json_output)
