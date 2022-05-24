""" Default parser for speaker representation """
import argparse
import os


def get_default_parser():
    parser = argparse.ArgumentParser(description="Encode speaker representations.")
    parser.add_argument(
        "--bdir",
        default="speaker-diarization",
        type=str,
        help="base directory for speaker diarization",
        dest="base_dir"
    )
    parser.add_argument(
        "--vdir",
        default=os.path.join("voice-activity-detection", "results", "speechbrain"),
        type=str,
        help="directory containing .rttm files with speech segments",
        dest="vad_dir"
    )
    parser.add_argument(
        "--max-length",
        default=20.0,
        type=float,
        help="maximum length of audio segments for encoding",
        dest="max_length"
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

    return parser
