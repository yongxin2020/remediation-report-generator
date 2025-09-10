"""
PredictedEmo_Analysis.py - Statistical Analysis of Predicted Emotions

This script performs statistical analysis of predicted emotions from cognitive remediation 
sessions by comparing participant data against the THERADIA test set baseline. It identifies 
salient emotions using z-score tests, Cohen's d effect size, and Bonferroni correction.

Key Functions:
- getLabelsFrequency()/getLabelsIntensity(): Extract emotion data from prediction files
- getSalientEmotions(): Statistical comparison with z-tests and effect size calculation
- analyze_participant_emotions(): High-level analysis comparing against test set subjects

Statistical Methods:
- Z-score computation for emotion intensity comparison
- Cohen's d effect size for practical significance assessment
- Bonferroni correction for multiple comparison control

Constants:
- THERADIA_SEQUENCE_COUNTS: Pre-defined sequence counts for 17 test sessions
- THERADIA_SESSION_NAMES: Ordered session identifiers

Usage:
    Can be used as a module or run standalone with example analysis workflows.
    Supports both frequency-based and intensity-based emotion analysis.

Dependencies: pandas, numpy, scipy.stats, ast, os
"""

import pandas as pd
import ast
import numpy as np
from numpy import mean
import os


def getLabelsFrequency(file):
    """Extract emotion frequency data from prediction file."""
    LabelsFrequency_TestSet = []
    TotalSequences_TestSet = []
    
    with open(file) as f:
        datafile = f.readlines()
    
    for line in datafile:
        if 'Frequency of affect labels for participant' in line:
            LabelsFrequency_TestSet.append(line)
        if 'Total sequences number of this session:' in line:
            total_sequences = line.split("Total sequences number of this session: ")[1]
            TotalSequences_TestSet.append(int(total_sequences))
    
    return LabelsFrequency_TestSet, TotalSequences_TestSet


def getLabelsIntensity(file):
    """Extract emotion intensity data from prediction file."""
    LabelsProbability_TestSet = []
    TotalSequences_TestSet = []
    
    with open(file) as f:
        datafile = f.readlines()
    
    for i, line in enumerate(datafile):
        if 'probability of each label {' in line:
            LabelsProbability_TestSet.append(line)
        elif 'probability of each label speechbrain.lobes.models.huggingface_wav2vec' in line:
            # Handle irregular format
            LabelsProbability_TestSet.append(datafile[i+4])
        elif 'Total sequences number of this session:' in line:
            total_sequences = line.split("Total sequences number of this session: ")[1]
            TotalSequences_TestSet.append(int(total_sequences))
    
    return LabelsProbability_TestSet, TotalSequences_TestSet


def preprocess_LabelsFrequency(LabelsFrequency_TestSet, TotalSequences_TestSet):
    """Convert emotion frequency data to DataFrame with session names."""
    sessions = []
    emotions = []

    for entry in LabelsFrequency_TestSet:
        # Extract session name and emotion dictionary
        session_info = entry.split(":")
        session_name = session_info[0].strip().split("Frequency of affect labels for participant ")[1]
        session_info_emo = entry.split(f"Frequency of affect labels for participant {session_name}: ")[1]
        
        # Parse emotion dictionary
        emotion_dict = ast.literal_eval(session_info_emo)
        
        sessions.append(session_name)
        emotions.append(emotion_dict)

    # Create DataFrame
    df_LabelsFrequency_TestSet = pd.DataFrame(emotions, index=sessions)
    df_LabelsFrequency_TestSet_new = df_LabelsFrequency_TestSet.assign(TotalSequences=TotalSequences_TestSet)
    
    return df_LabelsFrequency_TestSet_new


def preprocess_LabelsIntensity(LabelsProbability_TestSet):
    """Convert emotion intensity data to DataFrame."""
    emotions_intensity = []

    for entry in LabelsProbability_TestSet:
        if "probability of each label " in entry:
            seq_emo_intensity = entry.split("probability of each label ")[1]
        else:
            seq_emo_intensity = entry
        
        seq_emo_intensity_dict = ast.literal_eval(seq_emo_intensity)
        emotions_intensity.append(seq_emo_intensity_dict)

    return pd.DataFrame(emotions_intensity)

def compare_value_to_norm(value, norm):
    """Compare a value against a norm range and return directional indicator."""
    norm_left = float(norm.split(" - ")[0])
    norm_right = float(norm.split(" - ")[1])
    
    if norm_left <= value <= norm_right:
        return ""
    elif value < norm_left: 
        return "&#129047;"  # Leftwards Arrow
    elif value > norm_right:
        return "&#129045;"  # Rightwards Arrow

def compute_z_score(sample_df, population_df):
    """Compute z-score and effect size for emotion intensity comparison."""
    import numpy as np
    import scipy.stats as stats
    
    sample_mean = sample_df.mean()
    population_mean = population_df.mean()
    population_std = population_df.std()
    sample_size = sample_df.count()
    alpha = 0.05
    
    # Compute z-score
    z_score = (sample_mean - population_mean) / (population_std / np.sqrt(sample_size))
    
    # Critical Z-Score
    z_critical = stats.norm.ppf(1 - alpha)
    
    # P-Value
    p_value = 1 - stats.norm.cdf(z_score)
    
    # Cohen's d effect size
    d = cohend(sample_df, population_df)
    
    return z_score, z_critical, p_value, d

def cohend(d1, d2):
    """Calculate Cohen's d effect size for independent samples."""
    from numpy import mean, var
    from math import sqrt
    
    n1, n2 = len(d1), len(d2)
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = mean(d1), mean(d2)
    d = (u1 - u2) / s
    return d


def getSalientEmotions(emotion_labels, df_subject, df_testset, num_subjects):
    """Identify salient emotions by comparing subject data with test set."""
    salient_emotions = []
    cohen_d = []
    
    for emo_label in emotion_labels:
        z_score, z_critical, p_value, d = compute_z_score(
            df_subject[emo_label], df_testset[emo_label]
        )
        
        # Apply Bonferroni correction
        alpha = 0.05 / num_subjects
        if p_value < alpha:
            salient_emotions.append(emo_label)
            cohen_d.append(d)
    
    return {'salient_emotions': salient_emotions, "cohen_d": cohen_d}

def separate_sessions_by_subject(sessions):
    """Group sessions by subject ID."""
    subject_sessions = {}
    
    for session in sessions:
        subject_id = session.split('_')[0]
        
        if subject_id in subject_sessions:
            subject_sessions[subject_id].append(session)
        else:
            subject_sessions[subject_id] = [session]
    
    return subject_sessions


def separate_data_by_sessions(df, total_sequences):
    """Split DataFrame into groups based on session sequence counts."""
    groups = {}
    start_idx = 0
    
    for session, num in zip(THERADIA_SESSION_NAMES, total_sequences):
        groups[session] = df.iloc[start_idx:start_idx + num]
        start_idx += num
    
    return groups


def sort_dict_by_values(dict_data):
    """Sort dictionary by values in descending order."""
    keys = list(dict_data.keys())
    values = list(dict_data.values())
    sorted_indices = np.argsort(values)[::-1]
    return {keys[i]: values[i] for i in sorted_indices}

# THERADIA test set session sequence counts (hardcoded for current dataset)
THERADIA_SEQUENCE_COUNTS = [
    208, 258, 171, 181, 317, 240, 241, 273,
    157, 254, 298, 285, 249, 191, 192, 117, 212
]

# THERADIA test set session names (ordered to match sequence counts)
THERADIA_SESSION_NAMES = [
    'A04E_seance1', 'A04E_seance2', 'A03E_seance1', 'A03E_seance2',
    'A02E_seance1', 'A02E_seance2', 'A01E_seance1', 'A01E_seance2',
    'A02D_seance1', 'A02C_seance1', 'A02B_seance1', 'A02A_seance1',
    'A01D_seance1', 'A01C_seance1', 'A01B_seance1', 'M08E_seance1',
    'M09E_seance1'
]


def flatten(lst):
    """Flatten a list of lists."""
    return [item for sublist in lst for item in sublist]


def analyze_participant_emotions(df_subject, df_testset_file, participant_code, session_num):
    """
    Analyze participant emotions against THERADIA test set.
    
    Args:
        df_subject: DataFrame with participant emotion data
        df_testset_file: Path to test set CSV file
        participant_code: Participant identifier
        session_num: Session number
        
    Returns:
        dict: Sorted dictionary of salient emotions with counts
    """
    # Load test set data
    if os.path.exists(df_testset_file):
        df_testset = pd.read_csv(df_testset_file)
    else:
        raise FileNotFoundError(f"Test set file not found: {df_testset_file}")
    
    # Get emotion labels
    emotion_labels = df_subject.columns
    
    # Separate data by sessions
    groups = separate_data_by_sessions(df_testset, THERADIA_SEQUENCE_COUNTS)
    subject_sessions = separate_sessions_by_subject(THERADIA_SESSION_NAMES)
    num_subjects = len(subject_sessions.keys())
    
    salient_emotions_all = []
    
    # Compare against each subject in test set
    for subject in subject_sessions.keys():
        session_list = subject_sessions[subject]
        
        if len(session_list) == 1:
            df_testset_subject = groups[session_list[0]]
        elif len(session_list) == 2:
            df_1 = groups[session_list[0]]
            df_2 = groups[session_list[1]]
            df_testset_subject = pd.concat([df_1, df_2])
        else:
            continue
        
        results = getSalientEmotions(emotion_labels, df_subject, df_testset_subject, num_subjects)
        salient_emotions_all.append(results["salient_emotions"])
    
    # Count occurrences and sort
    all_emotions = flatten(salient_emotions_all)
    emotion_counts = {x: all_emotions.count(x) for x in set(all_emotions)}
    
    return sort_dict_by_values(emotion_counts)


if __name__ == '__main__':
    # ########################################
    # Different Analysis Options
    # ########################################
    
    # Option 1: Analysis by emotion frequency (labels)
    # Path to the file containing predictions of all 17 sessions on the test set
    # File_LabelsFrequency_TestSet = "../LabelsPredictions_TestSet_All.txt"
    # File_LabelsFrequency_Subject = "../M01E_seance1_0405.txt"

    # LabelsFrequency_TestSet, TotalSequences_TestSet = getLabelsFrequency(File_LabelsFrequency_TestSet)
    # LabelsFrequency_Subject, TotalSequences_Subject = getLabelsFrequency(File_LabelsFrequency_Subject)

    # df_LabelsFrequency_TestSet = preprocess_LabelsFrequency(LabelsFrequency_TestSet, TotalSequences_TestSet)
    # df_LabelsFrequency_Subject = preprocess_LabelsFrequency(LabelsFrequency_Subject, TotalSequences_Subject)
    # df_LabelsFrequency_TestSet.to_csv('./LabelsFrequency_TestSet.csv')
    
    # Load pre-computed frequency data
    # df_LabelsFrequency_TestSet = pd.read_csv("./LabelsFrequency_TestSet.csv")
    # sessions = df_LabelsFrequency_TestSet.iloc[:,0].tolist()
    # print(sessions)

    # Calculate ratios and compare with norms
    # df_LabelsFrequency_TestSet_RATIO = get_df_ratio(df_LabelsFrequency_TestSet, TotalSequences_TestSet)
    # df_LabelsFrequency_Subject_RATIO = get_df_ratio(df_LabelsFrequency_Subject, TotalSequences_Subject)

    # emotion_labels = df_LabelsFrequency_TestSet.columns

    # for emo_label in emotion_labels:
    #     norm = get_mean_std_each_variable(df_LabelsFrequency_TestSet_RATIO, emo_label)
    #     print(df_LabelsFrequency_Subject_RATIO[emo_label])
    #     comparison = compare_value_to_norm(df_LabelsFrequency_Subject_RATIO[emo_label].values[0], norm)
    #     print(f"{emo_label}'s frequency compared with the norm of test set:", comparison)

    # ########################################
    # Option 2.1: Analysis by emotion intensity (combined distribution)
    # ########################################
    
    # Load or generate intensity data for test set
    # if os.path.exists('./LabelsProbability_TestSet.csv'):
    #     df_LabelsIntensity_TestSet = pd.read_csv("./LabelsProbability_TestSet.csv")
    # else:
    #     LabelsProbability_TestSet, TotalSequences_TestSet_intensity = getLabelsIntensity(File_LabelsFrequency_TestSet)
    #     df_LabelsIntensity_TestSet = preprocess_LabelsIntensity(LabelsProbability_TestSet)
    #     df_LabelsIntensity_TestSet.to_csv('./LabelsProbability_TestSet.csv')

    # Process subject data
    # LabelsProbability_Subject, TotalSequences_Subject_intensity = getLabelsIntensity(File_LabelsFrequency_Subject)
    # df_LabelsIntensity_Subject = preprocess_LabelsIntensity(LabelsProbability_Subject)

    # Get emotion labels and analyze
    # emotion_labels = df_LabelsIntensity_Subject.columns
    # salientEmotions = getSalientEmotions(emotion_labels, df_LabelsIntensity_Subject, df_LabelsIntensity_TestSet)
    # print("salientEmotions", salientEmotions)

    # ########################################
    # Option 2.2: Analysis by emotion intensity (compared with individual subjects)
    # ########################################
    
    # subject_sessions = separate_sessions_by_subject(sessions)
    # num_subjects = len(subject_sessions.keys())
    # groups = separate_data_by_sessions(df_LabelsIntensity_TestSet, THERADIA_SEQUENCE_COUNTS)

    # salientEmotions_All = []
    # cohend_All = []
    # 
    # for subject in subject_sessions.keys():
    #     session_list = subject_sessions[subject]
    #     
    #     if len(session_list) == 1:
    #         df_LabelsIntensity_TestSet_by_subject = groups[session_list[0]]
    #     elif len(session_list) == 2:
    #         df_1 = groups[session_list[0]]
    #         df_2 = groups[session_list[1]]
    #         df_LabelsIntensity_TestSet_by_subject = pd.concat([df_1, df_2])
    #     else:
    #         continue
    #     
    #     EMOresults = getSalientEmotions(emotion_labels, df_LabelsIntensity_Subject,
    #                                     df_LabelsIntensity_TestSet_by_subject, num_subjects)
    #     salientEmotions_All.append(EMOresults["salient_emotions"])
    #     cohend_All.append(EMOresults["cohen_d"])
    #     
    #     print(f"Compared with {subject} in the test set, salient emotions of {code}_seance{seance}:",
    #           EMOresults["salient_emotions"], "with Cohen's d:", EMOresults["cohen_d"])
    # 
    # # Aggregate results across all subjects
    # salientEmotions_All_list = flatten(salientEmotions_All)
    # participant_salientEmotions_dict = {x: salientEmotions_All_list.count(x) for x in set(salientEmotions_All_list)}
    # sorted_dict = sort_dict_by_values(participant_salientEmotions_dict)
    # print("Final sorted salient emotions:", sorted_dict)

    # # Calculate average Cohen's d for each emotion
    # averages = {}
    # for i, emotions in enumerate(salientEmotions_All):
    #     if cohend_All[i]:  # Check if cohen_d list is not empty
    #         avg_cohen_d = mean(cohend_All[i])
    #         for emotion in emotions:
    #             if emotion not in averages:
    #                 averages[emotion] = []
    #             averages[emotion].append(avg_cohen_d)

    # for emotion, cohen_d_list in averages.items():
    #     average_cohen_d = mean(cohen_d_list)
    #     print(f"Average Cohen's d for {emotion}: {average_cohen_d}")

    # ########################################
    # Default: Simple module test
    # ########################################
    
    # Example usage for testing
    testset_file = "./LabelsProbability_TestSet.csv"
    
    if os.path.exists(testset_file):
        print("Test set file found. Module ready for use.")
        print("Available analysis functions:")
        print("  - getLabelsFrequency(): Extract frequency data")
        print("  - getLabelsIntensity(): Extract intensity data") 
        print("  - getSalientEmotions(): Compare subject vs test set")
        print("  - analyze_participant_emotions(): High-level analysis")
    else:
        print(f"Warning: Test set file not found at {testset_file}")
        print("Please ensure the test set data is available for emotion analysis.")
        print("Expected file: LabelsProbability_TestSet.csv")