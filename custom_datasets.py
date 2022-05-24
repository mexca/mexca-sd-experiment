from datasets import load_dataset
import os
import torchaudio


def load_ded21_dataset(filepath):
    audio_path = os.path.join(filepath, "ded21_audio.wav")
    signal, sample_rate = torchaudio.load(audio_path)
    dataset = [{
        "file": audio_path,
        "audio": {
            "array": signal,
            "sampling_rate": sample_rate
        }
    }]

    return dataset


def load_dataset_from_args(args_dataset):
    if args_dataset == "ami_micro_test":
        return load_dataset("ami", "microphone-single", split=["test"])[0]
    if args_dataset == "ded21":
        return load_ded21_dataset("dutch-debate-corpus")
    
    raise Exception("Please specify a valid dataset argument")


def get_file_indices(args):
    if args.files_is_range:
        file_indices = range(args.files[0], args.files[1])
    else:
        file_indices = args.files

    return file_indices
