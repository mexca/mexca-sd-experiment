""" Functions for encoding speaker representations for diarization """
from datasets import load_dataset
from rttm import read_rttm
import os
import torch


def load_dataset_from_args(args_dataset):
    if args_dataset == "ami_micro_test":
        return load_dataset("ami", "microphone-single", split=["test"])[0]
    else:
        raise Exception("Please specify a valid dataset argument")


def get_file_indices(args):
    if args.files_is_range:
        file_indices = range(args.files[0], args.files[1])
    else:
        file_indices = args.files

    return file_indices


def get_dir_label(dir_name):
    dir_labels = {
        "speechbrain-ecapa-tdnn": "sb_ecapa_tdnn",
        "transformers-unisat": "tr_unisat",
        "transformers-wavlm": "tr_wavlm"
    }

    return dir_labels[dir_name]


def load_speech_sequences(vad_dir):
    rttm_seqs = []

    with os.scandir(vad_dir) as filenames:
        for filename in filenames:
            rttm_seqs.append(read_rttm(filename.path))

    return rttm_seqs


def encode_speakers(data, args, pipeline, encode_fun):
    file_indices = get_file_indices(args)
    rttm_seqs = load_speech_sequences(args.vad_dir)

    for i, sample in enumerate(data):
        if i in file_indices:
            filename = sample["file"]
            sample_rate = sample["audio"]["sampling_rate"]
            rttm_seq = rttm_seqs[i]
            assert rttm_seq.sequence[0].file == filename
            audio_segments = rttm_seq.get_audio_segments(
                sample["audio"]["array"], sample_rate, max_length=args.max_length*sample_rate)
            embeddings = encode_fun(audio_segments, sample_rate)
            torch.save(embeddings, os.path.join(args.base_dir, "embeddings",
                       pipeline, f"{get_dir_label(pipeline)}_{args.dataset}_{i}.pt"))
            print(f"Processed sample {i}: {filename}")
