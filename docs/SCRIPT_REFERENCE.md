# Summary of Python Scripts
Below is a summary of the key Python scripts used in the project, including their objectives, inputs, and outputs.

| Directory               | Script                        | Objective                                                                 | Input                                      | Output                                   |
|-------------------------|-------------------------------|---------------------------------------------------------------------------|--------------------------------------------|------------------------------------------|
| `generate/`             | `dialogues_theradia.py`       | Extract information from dialogue transcripts (CSV).                       | CSV file with dialogue turns.              | Extracted linguistic features.           |
| `generate/`             | `logs_theradia.py`            | Extract session metadata and exercise results from log files.              | Log file with session and exercise data.   | Extracted session and exercise details.  |
| `generate/`             | `main_phrases_trous.py`       | Define cloze sentences and generate the report.                            | Extracted variables from all sources.      | HTML/Markdown report.                    |
| `generate/`             | `calculate_norme.py`          | Calculate linguistic norms for comparison.                                 | `User_Info_Stat.csv`                       | Calculated norms for the elder group.    |
| `emotion_analysis/`             | `emotion_pipeline.py`         | Pipeline of emotion prediction -> analysis -> reporting, used in main generator script.  | Preprocessed emotion predictions (CSV).    | Salient emotions for the report.         |
| `emotion_analysis/`  | `prediction_audio_video.py` | Perform emotion recognition from audio/video files.                     | Audio/video recordings.                    | Emotion predictions (CSV).               |
| `emotion_analysis/`     | `PredictedEmo_Analysis.py`    | Statistical analysis of predicted emotions against THERADIA test set.      | Emotion prediction files (.txt/.csv).      | Salient emotions and statistical metrics. |
| `data/`                 | `Dataset.py`                  | Define the THERADIA dataset class.                                         | Custom corpus data.                        | Adapted dataset class.                   |
| `data/`                 | `DataStore.py`                | Store session data in a JSON file.                                         | Session data (logs, transcripts, etc.).    | `data.json` file.                        |
| `statistics/`           | `descriptors_stat_csv.py`     | Create a CSV file with feature values.                                     | Session data and features.                 | `User_Info_Stat.csv`.                    |
