""" Encode speaker representations in AMI test data set using transformers UniSpeech-SAT """
from custom_datasets import load_dataset_from_args
from default_parser import get_default_parser
from speaker_representation import encode_speakers
from transformers import Wav2Vec2FeatureExtractor, UniSpeechSatForXVector
import torch

parser = get_default_parser()
args = parser.parse_args()

model = UniSpeechSatForXVector.from_pretrained("microsoft/unispeech-sat-base")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    'microsoft/wavlm-base-sv')

data = load_dataset_from_args(args.dataset)


def encode_segments(audio_segments, sample_rate):
    embeddings = torch.empty((len(audio_segments), 512))

    for j, segment in enumerate(audio_segments):
        inputs = feature_extractor(
            segment, sampling_rate=sample_rate, padding=True, do_normalize=True, return_tensors="pt")
        with torch.no_grad():
            embeddings[j, :] = model(**inputs).embeddings.squeeze()

    return embeddings


encode_speakers(data, args, "transformers-unisat", encode_segments)
