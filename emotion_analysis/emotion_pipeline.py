import os
import pandas as pd
from pathlib import Path
import yaml
import ast
from typing import Dict, List
from PredictedEmo_Analysis import (
    getSalientEmotions,
    separate_data_by_sessions,
    separate_sessions_by_subject,
    sort_dict_by_values,
    THERADIA_SEQUENCE_COUNTS
)


class EmotionAnalysisPipeline:
    def __init__(self, settings_path: str, participant: str, session: int):
        self.config = self._load_config(settings_path)
        self.participant = participant
        self.session = session
        self.test_set_intensity = None
        self.test_set_frequency = None

    def _load_config(self, settings_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(settings_path, 'r') as f:
            return yaml.safe_load(f)

    def load_test_set_data(self) -> None:
        """Load and preprocess test set data."""
        intensity_path = Path(self.config['paths']['emotion']['test_set_intensity'])
        frequency_path = Path(self.config['paths']['emotion']['test_set_frequency'])

        self.test_set_intensity = pd.read_csv(intensity_path)
        self.test_set_frequency = pd.read_csv(frequency_path)

    def organize_test_set(self) -> tuple:
        """Organize test set data by subjects and sessions."""
        sessions = self.test_set_frequency.iloc[:, 0].tolist()
        subject_sessions = separate_sessions_by_subject(sessions)
        session_groups = separate_data_by_sessions(
            self.test_set_intensity,
            THERADIA_SEQUENCE_COUNTS
        )
        return subject_sessions, session_groups

    def load_subject_data(self) -> pd.DataFrame:
        """Load participant's emotion data."""
        results_dir = Path(self.config['paths']['emotion']['results_dir'])
        file_path = results_dir / f"{self.participant}_seance{self.session}.csv"

        if not file_path.exists():
            print(f"Warning: No predictions found for {self.participant} session {self.session}")
            return None

        df = pd.read_csv(file_path)
        intensity_data = df['probs_all'].apply(ast.literal_eval)
        return pd.DataFrame(intensity_data.tolist())

    def analyze_emotions(self) -> Dict:
        """Run complete emotion analysis pipeline."""
        self.load_test_set_data()
        subject_sessions, session_groups = self.organize_test_set()
        subject_data = self.load_subject_data()

        if subject_data is None:
            return {'salient_emotions': {}, 'cohen_d_values': []}

        emotion_labels = subject_data.columns
        num_subjects = len(subject_sessions.keys())

        salient_emotions = []
        cohen_d_values = []

        for subject, sessions in subject_sessions.items():
            if len(sessions) == 1:
                test_set_df = session_groups[sessions[0]]
            else:
                test_set_df = pd.concat([
                    session_groups[sessions[0]],
                    session_groups[sessions[1]]
                ])

            results = getSalientEmotions(
                emotion_labels,
                subject_data,
                test_set_df,
                num_subjects
            )

            salient_emotions.append(results["salient_emotions"])
            cohen_d_values.append(results["cohen_d"])

        # Process results
        flattened_emotions = [
            emo for sublist in salient_emotions for emo in sublist
        ]
        emotion_counts = {
            emo: flattened_emotions.count(emo)
            for emo in set(flattened_emotions)
        }

        return {
            'salient_emotions': sort_dict_by_values(emotion_counts),
            'cohen_d_values': cohen_d_values
        }

    def run_analysis(self, audio_source: str, video_source: str) -> Dict:
        """Complete pipeline from raw data to analysis."""
        # First try to load existing predictions
        subject_data = self.load_subject_data()

        # If no predictions exist, run emotion prediction
        if subject_data is None:
            print(f"Running emotion prediction for {self.participant} session {self.session}")
            self.run_emotion_prediction(audio_source, video_source)
            subject_data = self.load_subject_data()
            if subject_data is None:
                return {'salient_emotions': {}, 'cohen_d_values': []}

        # Then analyze the emotions
        return self.analyze_emotions()

    def run_emotion_prediction(self, audio_source: str, video_source: str) -> None:
        """Run emotion prediction on audio/video files."""
        from prediction_audio_video import run_prediction

        # Create mock arguments object
        class Args:
            def __init__(self, id_participant, output_path):
                self.id_participant = id_participant
                self.output_path = output_path

        # Ensure output directory exists
        output_dir = self.config['paths']['emotion']['results_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Run prediction
        args = Args(
            id_participant=self.participant,
            output_path=output_dir
        )

        try:
            run_prediction(args)
        except Exception as e:
            print(f"Prediction failed: {e}")
            raise