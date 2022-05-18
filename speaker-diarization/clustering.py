""" Cluster speaker representations and assign speaker labels """
from types import new_class
from numpy import indices
from pyannote.core import Segment, Annotation
from rttm import read_rttm
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from speaker_representation import load_speech_sequence, get_file_indices
import argparse
import os
import torch

parser = argparse.ArgumentParser(
    description="Cluster speaker representations and assign speaker labels.")
parser.add_argument(
    "--base-dir",
    default="speaker-diarization",
    type=str,
    help="base directory for speaker diarization",
    dest="base_dir"
)
parser.add_argument(
    "--res-dirs",
    nargs="+",
    default=["speechbrain-ecapa-tdnn",
             "transformers-unisat", "transformers-wavlm"],
    type=str,
    help="list of pipeline directory names within the base directory",
    dest="res_dirs"
)
parser.add_argument(
    "--vad-dir",
    default=os.path.join("voice-activity-detection", "results", "speechbrain"),
    type=str,
    help="directory containing .rttm files with speech segments",
    dest="vad_dir"
)
parser.add_argument(
    "--pipeline-labels",
    nargs="+",
    default=["sb_ecapa_tdnn", "tr_unisat", "tr_wavlm"],
    type=str,
    help="list of labels for pipelines (must be same length as --res-dirs)",
    dest="pipeline_labels"
)
parser.add_argument(
    "--dataset",
    default="ami",
    type=str,
    help="dataset on which speaker diarization is performed",
    dest="dataset"
)
parser.add_argument(
    "--speaker-labels",
    nargs="+",
    default=["A", "B", "C", "D", "E"],
    type=str,
    help="list with speaker labels",
    dest="speaker_labels"
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


def get_dir_labels(res_dirs, pipeline_labels):
    dir_labels = {}

    for i, res_dir in enumerate(res_dirs):
        dir_labels[res_dir] = pipeline_labels[i]

    return dir_labels


def convert_rttm_annotation(rttm_seq):
    annotation = Annotation()
    for seg in rttm_seq.sequence:
        annotation[Segment(seg.tbeg, seg.tbeg+seg.tdur)] = seg.name

    return annotation


def average_subsegment_embeddings(rttm_seq, embeddings):
    splits = []

    for seg in rttm_seq.sequence:
        splits.append(seg.tdur // 20.0 + 1)

    new_embeddings = torch.empty((len(splits), embeddings.shape[1]))

    for j, n in enumerate(splits):
        new_embedding = torch.mean(embeddings[j:(j+int(n)), :], dim=0)
        new_embeddings[j, :] = torch.nn.functional.normalize(
            new_embedding, dim=-1).cpu()

    assert new_embeddings.shape[0] == len(splits)

    return new_embeddings


def load_speaker_embeddings(model, base_dir, index):
    with os.scandir(os.path.join(base_dir, "embeddings", model)) as filenames:
        for filename in filenames:
            if int(filename.name.split("_")[-1].split(".")[0]) == index:
                embeddings = torch.load(filename.path)
                return embeddings


def cluster_speaker_embeddings(args, classifier_labels):
    dir_labels = get_dir_labels(args.res_dirs, args.pipeline_labels)
    indices = get_file_indices(args)

    for model in args.res_dirs:
        for i in indices:
            embeddings = load_speaker_embeddings(model, args.base_dir, i)
            rttm_seq = load_speech_sequence(args.vad_dir, i)
            new_embeddings = average_subsegment_embeddings(rttm_seq, embeddings)

            for cluster_method in classifier_labels.keys():
                classifier = classifier_labels[cluster_method](
                    len(args.speaker_labels))
                num_labels = classifier.fit_predict(new_embeddings)
                spk_labels = [args.speaker_labels[i] for i in num_labels]

                for j, seg in enumerate(rttm_seq.sequence):
                    seg.name = spk_labels[j]

                rttm_seq.write(os.path.join(args.base_dir, "results", model,
                                            f"{cluster_method}_{dir_labels[model]}_{args.dataset}_sample_{i}.rttm"))


classifier_labels = {
    "sc": SpectralClustering,
    "ac": AgglomerativeClustering,
    "km": KMeans
}

cluster_speaker_embeddings(args, classifier_labels)
