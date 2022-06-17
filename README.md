# Mexca Speaker Diarization Experiment
A repository for comparing potential speaker diarization tools to be used in the MEXCA pipeline.

## Structure
The repository contains subdirectories for different parts of the experiment:
- `automatic-speech-recognition/`: Contains files for exploring automatic speech recognition in Dutch on the DED21 dataset
- `speaker-diarization/`: Contains all files for the speaker diarization part
    - `embeddings/`: Contains the encoded speaker embeddings as .pt files
    - `results/`: Contains the .rttm files with speaker annotations
    - `clustering.py`: Script for clustering the speaker embeddings and assigning the speaker labels to speaker segments
    - `compare_sd.ipynb`: Notebook for comparing the speaker diarization approaches
    - `default_parser.py`: Helper functions for argument parsing
    - `plot_pipelines_results.R`: Script for visualizing the pipeline comparison
    - `pyannote_sd_compare.ipynb`: Notebook for analyzing the results of the `pyannote.audio` pipeline
    - `sd_*.py`: Scripts for applying the respective speaker encoding models
    - `sd_pipeline_ded21_performance.ipynb`: Notebook for analyzing the results of the most promising pipelines on the DED21 dataset
    - `speaker_diarization.py`: Script to run all speaker encoding scripts after each other
    - `speaker_representation.py`: Helper functions for performing speaker diarization
- `speaker-segmentation/`: Contains all files for the speaker segmentation part
    - `results/`: Cotains the .rttm files with speech segments
    - `seg_pyannote.py`: Script for applying speaker segmentation using the `pyannote.audio` package
- `voice-activity-detection/`: Contains all files for the voice activity detection part
    - `results/`: Contains the .rttm files with speech segments
    - `compare_vad.ipynb`: Notebook for comparing the voice activity detection approaches
    - `custom.conf`: Configuration file for the opensmile feature extractor
    - `opensmile_helper_functions`: Helper functions for extracting opensmile voice activity features
    - `vad_*.py`: Scripts for applying the voice activity detection models
- `create_ded21_corpus.ipynb`: Notebook for creating and exploring the DED21 dataset
- `explore_ami_corpus.ipynb`: Notebook for exploring the properties of the AMI corpus
- `rttm.py`: Functions for creating, reading, modifying, and writing .rttm files and objects
- `rttm_test.py`: Preliminary test suite for `rttm.py`

## Method
We compare multiple pipelines for voice activity detection (VAD), speaker segmentation, speaker diarization, and explore automatic speech recognition. We apply these tools to one public dataset ([AMI](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml) corpus; single channel; microphone; test set) using the default parameters and minimal postprocessing steps. The second data set ([Dutch Election Debate 2021](https://stukroodvlees.nl/welke-lijsttrekkers-lacht-het-meest-en-hoe/)) is not openly available yet (due to copyright issues).

## Results
The results of our pipeline comparisons are shown in the notebooks:
- [compare_vad.ipynb](https://github.com/mexca/mexca-sd-experiment/blob/main/voice-activity-detection/compare_vad.ipynb)
- [compare_sd.ipynb](https://github.com/mexca/mexca-sd-experiment/blob/main/speaker-diarization/compare_sd.ipynb)
- [pyannote_sd_compare.ipynb](https://github.com/mexca/mexca-sd-experiment/blob/main/speaker-diarization/pyannote_sd_compare.ipynb)
- [sd_pipeline_ded21_performance.ipynb](https://github.com/mexca/mexca-sd-experiment/blob/main/speaker-diarization/sd_pipeline_ded21_performance.ipynb)

In the speaker diarization comparison, the `pyannote.audio` pipeline outperformed all other candidates, achieving an average diarization error rate of 0.32 on the AMI test set (0.21 without speech overlap) and [0.35, 0.38] on the two parts of the DED21 data set (no speech overlap).

## References
Bredin, H. et al. (2020). Pyannote.audio: Neural building blocks for speaker diarization. *ICASSP 2020*, pp. 7124-7128. [URL](https://doi.org/10.1109/ICASSP40776.2020.9052974)

Carletta, J. (2006). Announcing the AMI meeting corpus. *The ELRA Newsletter 11*(1), pp. 3-5. [URL](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)

Schumacher, G., Homan, M., & Pipal, C. (March, 2021). Welke dijsttrekker lacht het meest? En hoe? *Stuk Rood Vlees*. [URL](https://stukroodvlees.nl/welke-lijsttrekkers-lacht-het-meest-en-hoe/)
