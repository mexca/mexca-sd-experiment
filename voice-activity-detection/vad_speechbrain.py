""" Detect voice activation in AMI test set using speechbrain """
from datasets import load_dataset
from speechbrain.pretrained import VAD
from rttm import RttmSeq, RttmObj
import os

DIR = "voice-activity-detection"

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir=os.path.join(DIR, "vad-crdnn-libriparty"))

ami = load_dataset("ami", "microphone-single", split=["test"])[0]

for i, sample in enumerate(ami):
    filename = sample["file"]
    boundaries = VAD.get_speech_segments(filename)
    segments = [RttmObj(
        type = "SPEAKER",
        file = filename,
        chnl = 1,
        tbeg = float(bnd[0]),
        tdur = float(bnd[1]-bnd[0])
    ) for bnd in boundaries]
    RttmSeq(segments).write(os.path.join(DIR, "results", "speechbrain", f"sb_ami_micro_test_sample_{i}.rttm"))
    print(f"Processed sample {i}: {filename}")
    