# Text-Audio-Video based Emotion recognition for THERADIA report generation
Codes adapted from [demo_emotion_theradia](https://gitlab.com/sina.alisamir/demo_emotion_theradia): Real-time Multimodal Emotion Recognition for the THERADIA corpus [[1]](#1).

This code provides multimodal (video-audio-text) emotion recognition based on the THERADIA corpus, with steps for identifying salient emotions for automated report generation.

## Installation

```bash
# install python3.9+ on your machine for running the code on a file
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the scripts from a file

To run emotion recognition on a video file:

```bash
source env/bin/activate # activate the environment
python prediction_audio_video.py # run the test code
```

- Use different parts of `prediction_audio_video.py` selectively
- **Note**: Default feature extraction uses W2V2 (audio) and BERT (text). Switch to TF-IDF (text) and MFB (audio) via settings in the file.

## How it works?

Multimodal emotion recognition is divided into modules:

1. `backend_audio.py`: Audio-textual module for voice activity detection, feature extraction, and emotion recognition.
2. `backend_video.py`: Video module for face detection, feature extraction, and emotion recognition.
3. `backend_main.py`: Main module for decision fusion of modality outputs and JSON generation.

- **Note**: All modules run independently, emotion recognition per modality doesn't depend on others.

## Core emotions
10 predefined emotions:
- Negative: 'annoyed', 'anxious', 'desperate', 'frustrated', 'surprised' 
- Positive: 'confident', 'happy', 'interested', 'relaxed', 'satisfied'

## Reference
<a id="1">[1]</a> 
 H. Fournier et al., "THERADIA WoZ: An Ecological Corpus for Appraisal-Based Affect Research in Healthcare" in IEEE Transactions on Affective Computing, vol. 16, no. 03, pp. 2233-2244, July-Sept. 2025, doi: 10.1109/TAFFC.2025.3557465.