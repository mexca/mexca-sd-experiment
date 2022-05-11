""" Encode speaker representations in AMI test data set using transformers WavLM """
from default_parser import get_default_parser
from speaker_representation import load_dataset_from_args, encode_speakers
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch

parser = get_default_parser()
args = parser.parse_args()

model = WavLMForXVector.from_pretrained('microsoft/wavlm-base')

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


encode_speakers(data, args, "transformers-wavlm", encode_segments)