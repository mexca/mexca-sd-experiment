""" Detect voice activation in AMI test set using opensmile and speechbrain """
from datasets import load_dataset
from speechbrain.pretrained import VAD
from rttm import RttmSeq, RttmObj
from opensmile_helper_functions import detect_voice_activation_opensmile
import os
import torch

DIR = "voice-activity-detection"

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir=os.path.join(DIR, "vad-crdnn-libriparty"))


def detect_voice_activation_opensmile_speechbrain(filename):
    segments_opensmile = detect_voice_activation_opensmile(filename)
    segments_merged = VAD.merge_close_segments(torch.tensor(segments_opensmile))
    segments_long = VAD.remove_short_segments(segments_merged)

    return segments_long


ami = load_dataset("ami", "microphone-single", split=["test"])[0]

for i, sample in enumerate(ami):
    filename = sample["file"]
    boundaries = detect_voice_activation_opensmile_speechbrain(filename)
    segments = [RttmObj(
        type = "SPEAKER",
        file = filename,
        chnl = 1,
        tbeg = float(bnd[0]),
        tdur = float(bnd[1]-bnd[0])
    ) for bnd in boundaries]
    RttmSeq(segments).write(os.path.join(DIR, "results", "opensmile_speechbrain", f"os_sb_ami_micro_test_sample_{i}.rttm"))
    print(f"Processed sample {i}: {filename}")
