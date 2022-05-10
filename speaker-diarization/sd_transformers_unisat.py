
from datasets import load_dataset
from rttm import read_rttm
from transformers import Wav2Vec2FeatureExtractor, UniSpeechSatForXVector
import os
import torch

DIR = "speaker-diarization"

RTTM_DIR = os.path.join("voice-activity-detection", "results", "speechbrain")

MAX_LENGTH = 20

model = UniSpeechSatForXVector.from_pretrained("microsoft/unispeech-sat-base")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    'microsoft/wavlm-base-sv')

ami = load_dataset("ami", "microphone-single", split=["test"])[0]

for i, sample in enumerate(ami):
    filename = sample["file"]
    sample_rate = sample["audio"]["sampling_rate"]
    rttm_seq = read_rttm(os.path.join(
        RTTM_DIR, f"sb_ami_micro_test_sample_{i}.rttm"))
    assert rttm_seq.sequence[0].file == filename
    audio_segments = rttm_seq.get_audio_segments(
        sample["audio"]["array"], sample_rate, max_length=MAX_LENGTH*sample_rate)
    inputs = feature_extractor(audio_segments, sampling_rate=sample_rate,
                               padding=True, do_normalize=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).embeddings.squeeze()
    torch.save(embeddings, os.path.join(DIR, "embeddings",
               "transformers-unisat", f"tr_unisat_ami_micro_test_{i}.pt"))
    print(f"Processed sample {i}: {filename}")
