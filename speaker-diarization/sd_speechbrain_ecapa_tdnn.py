""" Encode speaker representations in AMI test data set using Speechbrains ECAPA-TDNN """
from custom_datasets import load_dataset_from_args
from default_parser import get_default_parser
from speaker_representation import encode_speakers
from speechbrain.pretrained import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor
import os
import torch

parser = get_default_parser()
args = parser.parse_args()

ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(args.base_dir, "ecapa-tdnn")
)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    'microsoft/wavlm-base-sv')

data = load_dataset_from_args(args.dataset)


def encode_segments(audio_segments, sample_rate):
    embeddings = torch.empty((len(audio_segments), 192))

    for j, segment in enumerate(audio_segments):
        inputs = feature_extractor(
            segment, sampling_rate=sample_rate, padding=True, do_normalize=True, return_tensors="pt")
        embeddings[j, :] = ecapa.encode_batch(
            inputs.input_values).squeeze()

    return embeddings


encode_speakers(data, args, "speechbrain-ecapa-tdnn", encode_segments)
