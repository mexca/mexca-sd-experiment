from custom_datasets import load_dataset_from_args, get_file_indices
from datasets import load_dataset
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from rttm import RttmSeq, RttmObj
import argparse
import os


parser = argparse.ArgumentParser(
    description="Detect speaker identity using pyannote.")
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

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

data = load_dataset_from_args(args.dataset)

file_indices = get_file_indices(args)

for i, sample in enumerate(data):
    if i in file_indices:
        filename = sample["file"]
        result = pipeline(filename)
        segments = [RttmObj(
            type="SPEAKER",
            file=filename,
            chnl=1,
            tbeg=float(seg.start),
            tdur=float(seg.end - seg.start),
            name=str(speaker)
        ) for seg, _, speaker in result.itertracks(yield_label=True)]        
        
        RttmSeq(segments).write(os.path.join(args.base_dir, "results",
                                             "pyannote", f"pa_{args.dataset}_sample_{i}.rttm"))
        print(f"Processed sample {i}: {filename}")
        
