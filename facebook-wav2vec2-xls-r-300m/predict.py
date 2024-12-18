import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model_name='facebook/wav2vec2-xls-r-300m'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    return processor, model

def predict(processor, model, audio_path):
    # Load an audio file
    speech_array, sampling_rate = torchaudio.load(audio_path)
    speech_array = speech_array.squeeze().numpy()

    # Process the audio
    features = processor(
        speech_array,
        sampling_rate=sampling_rate,
        return_tensors='pt',
        padding=True
    )

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits 

    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]

    return transcription