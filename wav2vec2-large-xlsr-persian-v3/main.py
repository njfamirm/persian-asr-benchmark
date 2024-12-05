import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and processor
model_name = 'm3hrdadfi/wav2vec2-large-xlsr-persian-v3'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

# Load an audio file
speech_array, sampling_rate = torchaudio.load('../assets/audio-01.wav')
speech_array = speech_array.squeeze().numpy()

# Prepare the features for the model
features = processor(
    speech_array, 
    sampling_rate=sampling_rate, 
    return_tensors='pt', 
    padding=True
)
input_values = features.input_values.to(device)
attention_mask = features.attention_mask.to(device)

# Predict the transcription of the audio file
with torch.no_grad():
    logits = model(input_values, attention_mask=attention_mask).logits
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(pred_ids[0])

# Print the transcription
print('Transcription:', transcription)