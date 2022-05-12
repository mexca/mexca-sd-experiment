""" Detect speaker segments in AMI test set using pyannote """
from datasets import load_dataset
from pyannote.audio.pipelines import SpeakerSegmentation
from rttm import RttmSeq, RttmObj
import argparse
import os

parser = argparse.ArgumentParser(
    description="Detect speaker segments using pyannote.")
parser.add_argument(
    "--bdir",
    default="speaker-segmentation",
    type=str,
    help="base directory for speaker segmentation",
    dest="base_dir"
)
parser.add_argument(
    "--dataset",
    default="ami_micro_test",
    type=str,
    help="dataset on which speaker diarization is performed",
    dest="dataset"
)
parser.add_argument(
    "--files-is-list",
    action="store_false",
    help="whether the files argument is a range or list of indices",
    dest="files_is_range"
)
parser.add_argument(
    "files",
    nargs="+",
    type=int,
    help="indices of files to be processed"
)

args = parser.parse_args()

HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.5, "offset": 0.5  #,
    # remove speech regions shorter than that many seconds.
    #"min_duration_on": 0.5,
    # fill non-speech regions shorter than that many seconds.
    #"min_duration_off": 0.5
}

pipeline = SpeakerSegmentation(segmentation="pyannote/segmentation", skip_stitching=True)
pipeline.instantiate(HYPER_PARAMETERS)

if args.dataset == "ami_micro_test":
    data = load_dataset("ami", "microphone-single", split=["test"])[0]
else:
    raise Exception("Please specify a valid dataset argument")

if args.files_is_range:
    file_indices = range(args.files[0], args.files[1])
else:
    file_indices = args.files


for i, sample in enumerate(data):
    if i in file_indices:
        filename = sample["file"]
        result = pipeline(filename)
        segments = [RttmObj(
            type="SPEAKER",
            file=filename,
            chnl=1,
            tbeg=float(seg.start),
            tdur=float(seg.end - seg.start)
        ) for seg in result.itersegments()]
        RttmSeq(segments).write(os.path.join(args.base_dir, "results",
                                             "pyannote", f"pa_ami_micro_test_sample_{i}.rttm"))
        print(f"Processed sample {i}: {filename}")
