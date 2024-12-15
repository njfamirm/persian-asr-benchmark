import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def load_model():
    model_name = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return processor, model

def predict(processor, model, audio_path):
    # Initialize the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
        language="fa"
    )

    result = pipe(audio_path)
    transcription = result["text"]
    return transcription
