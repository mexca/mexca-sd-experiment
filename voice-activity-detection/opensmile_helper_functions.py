""" Helper functions for VAD with opensmile """
import opensmile
import os

DIR = "voice-activity-detection"

def extract_voice_prob_opensmile(filename):
    """ Extracts voice probability using opensmile """
    extractor_custom = opensmile.Smile(
        feature_set=os.path.join(DIR, "custom.conf"),
        feature_level="pitch_acf"
    )

    voice_prob = extractor_custom.process_file(filename)["voiceProb"]

    return voice_prob


def create_voice_activation_boundaries_opensmile(voice_prob):
    """ Creates speech segment boundaries from opensmile voice probabilities """
    segments = []
    tmp = False
    for i, frame in voice_prob.reset_index().iterrows():
        time = frame["start"].total_seconds()
        if frame["voiceProb"] >= 0.5 and not tmp:
            tmp = True
            segments.append([time])
        elif frame["voiceProb"] < 0.25 and tmp:
            tmp = False
            segments[-1].append(time)

    if len(segments[-1]) == 1:
        segments[-1].append(time)

    return segments


def detect_voice_activation_opensmile(filename):
    voice_prob = extract_voice_prob_opensmile(filename)
    segments = create_voice_activation_boundaries_opensmile(voice_prob)

    return segments