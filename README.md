# mexca-sd-experiment
A repository for comparing potential speaker diarization tools to be used in the MEXCA pipeline.

## Structure
The repository contains subdirectories for different parts of the experiment:
- `speaker-diarization\`: Contains all files for the speaker diarization part
    - `embeddings\`: Contains the encoded speaker embeddings as .pt files
    - `results\`: Contains the .rttm files with speaker annotations
    - `clustering.py`: Script for clustering the speaker embeddings and assigning the speaker labels to speaker segments
    - `sd_*.py`: Scripts for applying the respective speaker encoding models
    - `compare_sd.ipynb`: Notebook for comparing the speaker diarization approaches
    - `speaker_diarization.py`: Script to run all speaker encoding scripts after each other
    - `speaker_representation.py`: Helper functions for performing speaker diarization
- `voice-activity-detection\`: Contains all files for the voice activity detection part
    - `results\`: Contains the .rttm files with speech segments
    - `compare_vad.ipynb`: Notebook for comparing the voice activity detection approaches
    - `custom.conf`: Configuration file for the opensmile feature extractor
    - `opensmile_helper_functions`: Helper functions for extracting opensmile voice activity features
    - `vad_*.py`: Scripts for applying the voice activity detection models
- `explore_ami_corpus.ipynb`: Notebook for exploring the properties of the AMI corpus
- `rttm.py`: Functions for creating, reading, modifying, and writing .rttm files and objects
- `rttm_test.py`: Preliminary test suite for `rttm.py`