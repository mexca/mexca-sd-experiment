""" Detect voice activation in AMI test set using opensmile """
from datasets import load_dataset
from opensmile_helper_functions import detect_voice_activation_opensmile
from rttm import RttmSeq, RttmObj
import os

DIR = "voice-activity-detection"

ami = load_dataset("ami", "microphone-single", split=["test"])[0]

for i, sample in enumerate(ami):
    filename = sample["file"]
    boundaries = detect_voice_activation_opensmile(filename)
    segments = [RttmObj(
        type = "SPEAKER",
        file = filename,
        chnl = 1,
        tbeg = float(bnd[0]),
        tdur = float(bnd[1]-bnd[0])
    ) for bnd in boundaries]
    RttmSeq(segments).write(os.path.join(DIR, "results", "opensmile", f"os_ami_micro_test_sample_{i}.rttm"))
    print(f"Processed sample {i}: {filename}")
