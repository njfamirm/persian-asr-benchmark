from predict import load_model, predict
from metrics import calculate_wer
from normalizer import normalizer

processor, model = load_model()

audio_path = '../assets/audio-01.wav'
ground_truth_path = '../assets/audio-01.txt'

# Read the ground truth text
with open(ground_truth_path, 'r') as file:
    ground_truth = file.read()

transcription = predict(processor, model, audio_path)

# Calculate WER before normalizing
wer_score_before = calculate_wer(ground_truth, transcription)

# Normalize the transcription
normalized_transcription = normalizer({"sentence": transcription})["sentence"]

# Calculate WER after normalizing
wer_score_after = calculate_wer(ground_truth, normalized_transcription)

print('Transcription:', transcription)
print('WER before normalization:', wer_score_before)
print('Normalized Transcription:', normalized_transcription)
print('WER after normalization:', wer_score_after)
