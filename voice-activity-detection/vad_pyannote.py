""" Detect voice activation in AMI test set using pyannote """
from datasets import load_dataset
from pyannote.audio import Pipeline
from rttm import RttmSeq, RttmObj
import os

DIR = "voice-activity-detection"

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")

ami = load_dataset("ami", "microphone-single", split=["test"])[0]

for i, sample in enumerate(ami):
    filename = sample["file"]
    result = pipeline(filename)
    segments = [RttmObj(
        type = "SPEAKER",
        file = filename,
        chnl = 1,
        tbeg = float(seg.start),
        tdur = float(seg.end - seg.start)
    ) for seg in result.itersegments()]
    RttmSeq(segments).write(os.path.join(DIR, "results", "pyannote", f"pa_ami_micro_test_sample_{i}.rttm"))
    print(f"Processed sample {i}: {filename}")
