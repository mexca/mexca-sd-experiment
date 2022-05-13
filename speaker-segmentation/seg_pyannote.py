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
    "onset": 0.5, "offset": 0.5,  
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.5,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.5
}

pipeline = SpeakerSegmentation(segmentation="pyannote/segmentation")
pipeline.instantiate(HYPER_PARAMETERS)

if args.dataset == "ami_micro_test":
    data = load_dataset("ami", "microphone-single", split=["test"])[0]
else:
    raise Exception("Please specify a valid dataset argument")

if args.files_is_range:
    file_indices = range(args.files[0], args.files[1])
else:
    file_indices = args.files

""" commented out
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
"""

# 13/05/2022 latest edit

# function to filter out segments whose duration is below a threshold
def duration_selection(segments, threshold):
    segments_filtered = [seg for seg in segments
       if seg.tdur > threshold] 
    return segments_filtered

# function to merge segments whose intersegment interval is below a threshold
def merge_segments(segments, threshold):
    merged_segments = []
    for index, segment in enumerate(segments[:-1]): # from first to second-to-last
        end = (segment.tbeg + segment.tdur) # end of first segment
        merge = (segments[index+1].tbeg - end) > threshold
        if merge: 
            joint_segment = RttmObj(
                type=segment.type,
                file=segment.file,
                chnl=segment.chnl,
                tbeg=float(segment.tbeg), #beginning of fist segment
                tdur=float(segment.tdur + segments[index+1].tdur) #duration 1st segment + duration 2nd segment
            )
            merged_segments.append(joint_segment)
    return merged_segments

# threshold arguments
filter_th = 0.5 # segment duration
merge_th = 0.1 # segment gap 

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
        
        filtered_segments = duration_selection(segments, filter_th) # remove short segments
        merged_segments = merge_segments(seg_filtered, merge_th) # merge close segments
        RttmSeq(merged_segments).write(os.path.join(args.base_dir, "results",
                                             "pyannote", f"pa_ami_micro_test_sample_{i}.rttm"))
        print(f"Processed sample {i}: {filename}")
    

        
