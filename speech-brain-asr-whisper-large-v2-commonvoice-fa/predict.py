
from speechbrain.inference.ASR import WhisperASR

def load_model():
    asr_model_name = "speechbrain/asr-whisper-large-v2-commonvoice-fa"
    asr_model = WhisperASR.from_hparams(source=asr_model_name, savedir="pretrained_models/model")
    return asr_model

def predict(asr_model, audio_path):
    transcription = asr_model.transcribe_file(audio_path)
    return transcription