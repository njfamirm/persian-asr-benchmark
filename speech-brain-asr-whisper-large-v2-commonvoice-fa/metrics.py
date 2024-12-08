from jiwer import wer

def calculate_wer(ground_truth, transcription):
    wer_score = wer(ground_truth, transcription)
    return wer_score