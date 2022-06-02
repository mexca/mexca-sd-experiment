""" Functions for encoding speaker representations for diarization """
from custom_datasets import get_file_indices
from rttm import read_rttm
import os
import torch


def get_dir_label(dir_name):
    dir_labels = {
        "speechbrain-ecapa-tdnn": "sb_ecapa_tdnn",
        "transformers-unisat": "tr_unisat",
        "transformers-wavlm": "tr_wavlm"
    }

    return dir_labels[dir_name]


def get_vad_label(vad_name):
    vad_labels = {
        os.path.join("voice-activity-detection", "results", "speechbrain"): "sb",
        os.path.join("voice-activity-detection", "results", "pyannote"): "pa",
        os.path.join("speaker-segmentation", "results", "pyannote"): "pa-seg"
    }

    return vad_labels[vad_name]


def load_speech_sequence(vad_dir, dataset, index):
    with os.scandir(vad_dir) as filenames:
        for filename in filenames:
            if int(filename.name.split("_")[-1].split(".")[0]) == index and filename.name.find(dataset):
                rttm_seq = read_rttm(filename.path)
                return rttm_seq


def encode_speakers(data, args, pipeline, encode_fun):
    file_indices = get_file_indices(args)

    for i, sample in enumerate(data):
        if i in file_indices:
            filename = sample["file"]
            sample_rate = sample["audio"]["sampling_rate"]
            rttm_seq = load_speech_sequence(args.vad_dir, args.dataset, i)
            assert os.path.split(os.path.normpath(rttm_seq.sequence[0].file))[-1] == os.path.split(os.path.normpath(filename))[-1]
            audio_segments = rttm_seq.get_audio_segments(
                sample["audio"]["array"], sample_rate, max_length=args.max_length*sample_rate)
            embeddings = encode_fun(audio_segments, sample_rate)
            torch.save(embeddings, os.path.join(args.base_dir, "embeddings",
                       pipeline, f"{get_dir_label(pipeline)}_{get_vad_label(args.vad_dir)}_{args.dataset}_{i}.pt"))
            print(f"Processed sample {i}: {filename}")
