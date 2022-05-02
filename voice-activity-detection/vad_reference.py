
from datasets import load_dataset
from rttm import RttmSeq, RttmObj
import os

DIR = "voice-activity-detection"

ami = load_dataset("ami", "microphone-single", split=["test"])[0]

for i, sample in enumerate(ami):
    filename = sample["file"]
    segments = [RttmObj(
        type = "SPEAKER",
        file = sample["file"],
        chnl = 1,
        tbeg = float(sample["segment_start_times"][i]),
        tdur = float(sample["segment_end_times"][i]-sample["segment_start_times"][i]),
        name = sample["segment_speakers"][i]
    ) for i, _ in enumerate(sample["segment_ids"])]
    RttmSeq(segments).write(os.path.join(DIR, "results", "reference", f"os_sb_ami_micro_test_sample_{i}.rttm"))
    print(f"Processed sample {i}: {filename}")
