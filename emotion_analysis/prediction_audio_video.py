"""
Test Set used to train multimodal ER models:
#subjects_test_list = ["A04E", "A03E", "A02E", "A01E", "A02D", "A02C", "A02B", "A02A", "A01D", "A01C", "A01B", "M08E", "M09E"]
Full CSV results of the test set saved in './emoReco_MRG_archived/ER_results_test_set/'
Full CSV results of the reporting samples (M01E-M07E) saved in './emoReco_MRG_archived/ER_results_samples/'
"""

import cv2, time, os
import glob
import wave
import pandas as pd
import numpy as np
from backend_video import BackendVideo
from datetime import datetime
import speechbrain as sb
import speech_recognition
from backend_audio import Backend_audio
from backend_main import Backend_main
import yaml
import argparse
from pathlib import Path


def run_prediction(args):
    """Main prediction function that can be called programmatically."""
    config = load_config()
    participant = args.id_participant
    output_dir = args.output_path

    group = participant[0]
    os.makedirs(output_dir, exist_ok=True)

    participant_audio_path = config['paths']['participant']['audio'].format(
        participant=participant)  # for example, A04E has two sessions: A04E_seance1_1011 and A04E_seance2_1711
    print("participant_audio_path:", participant_audio_path)
    num_folders = count_folders(participant_audio_path)
    print("number of seances:", num_folders)

    total_sequences_test_set = 0
    emo_present_subjects_test = []

    for num in range(num_folders):
        num_seance = num + 1
        RuntimeError_Counts = 0
        ValueError_Counts = 0
        audio_files, video_files = get_wav_paths(participant, group, num_seance, config)
        total_sequences = len(audio_files)
        total_sequences_test_set = total_sequences_test_set + total_sequences
        print("Total sequences number of this session:", total_sequences)
        emo_present = []
        trs_list = []
        probs_all_list = []
        for i in range(len(audio_files)):
            wav_path = audio_files[i]
            video_path = video_files[i]
            try:
                trs, keys_with_value_1, probs_all = EMO_THERADIA(wav_path, video_path)
                trs_list.append(trs)
                emo_present.append(keys_with_value_1)
                probs_all_list.append(probs_all)

            except RuntimeError:
                print(f"RuntimeError for {video_path}")
                RuntimeError_Counts += 1
                trs = ""
                keys_with_value_1 = ""
                probs_all = ""
                trs_list.append(trs)
                emo_present.append(keys_with_value_1)
                probs_all_list.append(probs_all)

            except ValueError:
                print(f"ValueError for {video_path}")
                ValueError_Counts += 1
                trs = ""
                keys_with_value_1 = ""
                probs_all = ""
                trs_list.append(trs)
                emo_present.append(keys_with_value_1)
                probs_all_list.append(probs_all)

        print("Occurrences of RuntimeError in this session:", RuntimeError_Counts)
        print("Occurences of ValueError in this session:", ValueError_Counts)

        # dictionary of lists
        participant_dict = {'audio_file': audio_files, 'trs': trs_list, 'emo_lables': emo_present,
                            'probs_all': probs_all_list}

        df = pd.DataFrame(participant_dict)

        # Save with proper path handling
        output_file = os.path.join(output_dir, f"{participant}_seance{num_seance}.csv")
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

        emo_present_list = flatten(
            emo_present)  # a list contains the whole affect labels predicted for all sequences of a subject
        participant_emo_frequency_dict = {x: emo_present_list.count(x) for x in set(emo_present_list)}
        print(f"Frequency of affect labels for participant {participant}_seance{num_seance}:",
              participant_emo_frequency_dict)  # {'anxious': 1, 'desperate': 1}

        emo_present_subjects_test.extend(emo_present_list)

    emo_frequency_dict = {x: emo_present_subjects_test.count(x) for x in set(emo_present_subjects_test)}
    print("Frequency of affect labels for test set:", emo_frequency_dict)  # {'anxious': 1, 'desperate': 1}

    print("Total number of sequences in the test set:", total_sequences_test_set)

    return True  # Return success status

def load_config():
    with open('../settings.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_numeric_part(filename):
    return int(filename.split("_")[-1].split(".")[0])

def get_wav_paths(participant, group, num_seance, config):
    """Get paths to audio and video files for a participant session.

    Args:
        participant: Participant ID
        group: Group identifier (M, A, J)
        num_seance: Session number
        config: Configuration dictionary

    Returns:
        tuple: (list of audio paths, list of video paths)
    """
    # Get paths
    audio_path_participant = config['paths']['participant']['audio'].format(participant=participant)
    session_paths = glob.glob(os.path.join(audio_path_participant, f"{participant}_seance{num_seance}_*"))

    if not session_paths:
        raise FileNotFoundError(f"No session found for participant {participant} seance {num_seance}")

    audio_path = session_paths[0]
    wav_files = glob.glob(os.path.join(audio_path, "*.wav"))
    # Sort the files based on the numeric part of the file name
    sorted_wav_files = sorted(wav_files, key=get_numeric_part)

    video_path = config['paths']['participant']['video'].format(
        group=group,
        participant=participant,
        num_seance=num_seance
    )
    video_files = glob.glob(os.path.join(video_path, "*.mp4"))
    sorted_video_files = sorted(video_files, key=get_numeric_part)

    # Apply test mode limitation if enabled
    if config['options'].get('test_mode', False):
        max_files = config.get('max_test_files', 2)
        return sorted_wav_files[:max_files], sorted_video_files[:max_files]

    return sorted_wav_files, sorted_video_files

# with wave.open(wav_path, "rb") as wave_file:
#     frame_rate_file = wave_file.getframerate()
#     print("frame rate:", frame_rate_file)

def flatten(xss):
    return [x for xs in xss for x in xs]

def EMO_THERADIA(wav_path, video_path):
    ### --------------------------------------
    ### Settings
    file_dir = os.path.dirname(os.path.abspath(__file__))  # this python file's directory
    settings = {
        "session_id": "user_0_session_0",
        "device": "cpu",
        "logger": False,
        "save_wav": False,
        "features": {
            "text": "bert_sentiment",  # tfidf | tfidf_google | bert_sentiment | bert_sentiment_google
            "audio": "w2v2_xlsr",  # mfb_normed | w2v2_xlsr
            "video": "clip",
        },
    }

    ### --------------------------------------
    ### Initialisations
    backend_video = (
        BackendVideo()
    )  # initialising the module for emotion recognition from video
    backend_AT = Backend_audio(
        settings=settings
    )  # initialising the module for emotion recognition from audio and its transcriptions
    backend_VAT = (
        Backend_main()
    )  # initialising the module for decision fusion of audio and video emotion recognition modules

    ### --------------------------------------
    ### Loading a sample video file (and conversion for audio)
    #video_path = os.path.join(file_dir, "data", "test.mp4")
    video = cv2.VideoCapture(video_path)
    # cap = video.isOpened()

    #wav_path = os.path.join(file_dir, "data", "test.wav") #test_tmp
    arg = f'ffmpeg -i {video_path} -ar 16000 -ac 1 -c:a pcm_s16le -af "volume=0dB" -hide_banner -v 0 -y {wav_path}'
    os.system(arg)
    waves_th = sb.dataio.dataio.read_audio(wav_path)  # .unsqueeze(0)
    # if "w2v2" in settings["features"]["audio"]:
    #     waves_th = waves_th.unsqueeze(0)
    waves_np = waves_th.numpy()

    ### -------------------------------------
    ### Features extraction for every (frame_freq) frames

    frame_freq = 10  # how many frames per second
    frame_id = 0
    buffer = []
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break
        frame_id += 1
        if frame_id % frame_freq == 0:
            frame_resized = cv2.resize(frame, (640, 480))
            current_time = time.time()
            datetime_str = datetime.fromtimestamp(current_time).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

            feats, _ = backend_video.get_features(frame_resized, frame_id, datetime_str)
            buffer.append(feats)

    ### --------------------------------------------------------
    ### Emotion recognition and dimensions prediction from video

    # selected emotions
    emo_indices = [2, 3, 4, 5, 7, 8]
    dimensions = [
        "activation",
        "adaptation",
        "conductivité au but",
        "plaisir_intrinsèque",
        "nouveauté",
    ]
    emotion_labels = ["confiant", "désespéré", "frustré", "heureux", "détendu", "satisfait"]

    dim_sum, cont_dim, labels = backend_video.get_predictions(np.array(buffer))
    video_preds = [[int(a * 100) for a in arr] for arr in labels]
    labels = np.array(labels)[emo_indices]

    dim_sum_dict = dict(zip(dimensions, np.array(dim_sum)))
    cont_dim_dict = dict(zip(dimensions, np.array(cont_dim)))
    labels_dict = dict(zip(emotion_labels, labels))

    print("dim_sum: {}".format(dim_sum_dict))
    print("cont_dim: {}".format(cont_dim_dict))
    print("labels: {}".format(labels_dict))

    ### --------------------------------------
    ### Speech recognition (can be replaced by any other)

    sr = speech_recognition.Recognizer()
    wav_bytes = (waves_np * 32767).astype(np.int16).tobytes()
    audio_data = speech_recognition.AudioData(wav_bytes, sample_rate=16000, sample_width=2)
    try:
        trs = sr.recognize_google(audio_data, language="fr-FR")
    except:
        trs = ""
    print("transcription:", trs)

    ### --------------------------------------
    ### Emotion recognition from speech and text

    preds_T, preds_T_dims_sum = backend_AT.predict_models_tfidf(
        trs, dir=settings["features"]["text"]
    )
    print("preds from text", preds_T)
    print("preds_dims_sum from text", preds_T_dims_sum)

    preds_A, preds_A_dims_sum = backend_AT.predict_models_audio(
        waves_np, dir=settings["features"]["audio"]
    )
    print("preds from audio", preds_A)
    print("preds_dims_sum from audio", preds_A_dims_sum)

    ### --------------------------------------
    ### Label decision fusion for video, audio, and text
    audio_preds = [preds_A[label] for label in backend_VAT.affect_labels]
    text_preds = [preds_T[label] for label in backend_VAT.affect_labels]
    print("audio_preds", audio_preds)
    print("text_preds", text_preds)
    print("video_preds", video_preds)
    text_audio_video_cls, probs_all, thresholds = backend_VAT.classify_preds(
        [text_preds, audio_preds, video_preds],
        [
            settings["features"]["text"],
            settings["features"]["audio"],
            settings["features"]["video"],
        ],
    )
    print(
        "each label prediction, and its influence from each modality", text_audio_video_cls
    ) # what I need
    print("probability of each label", probs_all)
    print("thresholds for choosing each label based on the probabilites", thresholds)

    ### --------------------------------------
    ### Dimension summaries decision fusion for video, audio, and text
    dim_details = {}
    for text_pred, audio_pred, video_pred, dim in zip(
        preds_T_dims_sum,
        preds_A_dims_sum,
        dim_sum,
        backend_VAT.dimensions,
    ):
        pred = (1 * text_pred + 1 * audio_pred + 1 * video_pred) / 3
        dim_details[dim] = round(pred, 3)

    print("dimension predictions", dim_details)

    keys_with_value_1 = [key for key, value in text_audio_video_cls.items() if value == 1]
    print(keys_with_value_1)  # ['happy', 'interested', 'relaxed']
    return trs, keys_with_value_1, probs_all

def count_folders(directory):
    folder_count = 0
    for _, dirs, _ in os.walk(directory):
        folder_count += len(dirs)
    return folder_count

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process participant data.')
    parser.add_argument('--id_participant', type=str, required=True,
                       help='Participant ID to process (e.g., A01E)')
    parser.add_argument('--output_path', type=str, default='./emoReco_MRG/',
                      help='Path to save output CSV files (default: ./emoReco_MRG/)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    run_prediction(args)
