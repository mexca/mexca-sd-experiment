
from custom_datasets import load_dataset_from_args
from rttm import RttmSeq, RttmObj
import argparse
import os

parser = argparse.ArgumentParser(description="Encode speaker representations.")
parser.add_argument(
    "--bdir",
    default="speaker-diarization",
    type=str,
    help="base directory for speaker diarization",
    dest="base_dir"
)
parser.add_argument(
    "--dataset",
    default="ami_micro_test",
    type=str,
    help="dataset on which speaker diarization is performed",
    dest="dataset"
)
args = parser.parse_args()

data = load_dataset_from_args(args.dataset)

for i, sample in enumerate(data):
    filename = sample["file"]
    segments = [RttmObj(
        type="SPEAKER",
        file=sample["file"],
        chnl=1,
        tbeg=float(sample["segment_start_times"][j]),
        tdur=float(sample["segment_end_times"][j] -
                   sample["segment_start_times"][j]),
        name=sample["segment_speakers"][j]
    ) for j, _ in enumerate(sample["segment_ids"])]
    RttmSeq(segments).write(os.path.join(args.base_dir, "results",
                                         "reference", f"ref_{args.dataset}_sample_{i}.rttm"))
    print(f"Processed sample {i}: {filename}")
