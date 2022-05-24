""" Run multiple speaker diarization pipelines """
import argparse
import os
import subprocess

parser = argparse.ArgumentParser(
    description="Run multiple speaker diarization pipelines.")
parser.add_argument(
    "--bdir",
    default="speaker-diarization",
    type=str,
    help="base directory for speaker diarization",
    dest="base_dir"
)
parser.add_argument(
    "--vdir",
    default=os.path.join("voice-activity-detection",
                         "results", "speechbrain"),
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

args = parser.parse_args()

subprocess.run([
    "python", os.path.join("speaker-diarization", "sd_speechbrain_ecapa_tdnn.py"),
    "--bdir", args.base_dir,
    "--vdir", args.vad_dir,
    "--max-length", str(args.max_length),
    "--dataset", args.dataset,
    "0", "16"],
    shell=True
)

subprocess.run([
    "python", os.path.join("speaker-diarization", "sd_transformers_unisat.py"),
    "--bdir", args.base_dir,
    "--vdir", args.vad_dir,
    "--max-length", str(args.max_length),
    "--dataset", args.dataset,
    "0", "16"],
    shell=True
)

subprocess.run([
    "python", os.path.join("speaker-diarization", "sd_transformers_wavlm.py"),
    "--bdir", args.base_dir,
    "--vdir", args.vad_dir,
    "--max-length", str(args.max_length),
    "--dataset", args.dataset,
    "0", "16"],
    shell=True
)
