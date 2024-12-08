import nemo.collections.asr as nemo_asr
import torch

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    asr_model_name = "nvidia/stt_fa_fastconformer_hybrid_large"
    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=asr_model_name).to(device)
    return asr_model

def predict(asr_model, audio_path):
    transcription = asr_model.transcribe([audio_path])[0]
    return transcription