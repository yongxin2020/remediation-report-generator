"""
This file is used to create a CSV file containing all the values of different features.
--> "User_Info_Stat.csv"
"""

import sys
# Insert the path of modules folder
sys.path.insert(0, "../generate/")

from main_phrases_trous import read_from_dialogue_file_to_log_file
from emoReco_theradia import getInfoFromMultimodal
# from DataStore import from_dialogue_file_find_log_file => this function has been commented out
import pandas as pd
import os
import glob

""" # old paths in 12/2022
path = "/data/depot/THERADIA-WoZ/Transcriptions"
dialogue_files = glob.glob(os.path.join(path, "*.csv")) # len(dialogue_files) # 129
# dialogue_files = [file for file in os.listdir(path) if file.endswith('.csv')] # file name without path, ex: data_agés_A__A01A_seance1_1512.csv

log_path_pattern= '/data/depot/THERADIA-WoZ/{}/{}/{}/'
"""
import yaml

path_to_yaml = "../settings.yaml"

with open(path_to_yaml, 'r') as file:
    config = yaml.safe_load(file)

mainPath = config["mainPath"]

def create_dataframe(data):
    df = pd.DataFrame(data, columns=['user_id', 'id_seance_num', 'age_group', 'induction_group',
                                     'date_session_string', 'textual_start_time', 'nb_activités', 'nb_exercices', 'duration_session_str',
                                     'start_time', 'duration_session',
                                     'num_réussi', 'num_moyen', 'num_raté',
                                     'taux_réussite',
                                     'réponse_affect_start', 'rep_long_courte',
#                                      'num_questions', 'questions_posées',
                                     'num_tokens_unique',
                                     'sum_duration_speak', 'débit_parole',
                                     'num_token_per_utterance_mean', 'num_token_per_utterance_max', 'num_token_per_utterance_min',
                                     'duration_utterance_mean', 'duration_utterance_max', 'duration_utterance_min',
                                     'lexical_diversity_TTR', 'lexical_diversity_BI', 'lexical_diversity_HS',
                                     'c_density_score', 'p_density_score'])

    return df


def main():
    data =[]
    #groups = ["A", "J", "M"]
    groups = ["A"]

    for group in groups:
        groupsPath = os.path.join(mainPath, f"{group}-subjects")
        print("groupsPath:", groupsPath)
        # transcriptionPath: / data / depot / processed - theradia - data / M - subjects / M01E / M01E_seance1_0405 / Transcripts and Logs
        csvFiles = glob.glob(os.path.join(groupsPath, "*/*/Transcripts and Logs/*_audio_farfield_trs.csv")) # list of paths
        print(csvFiles[:2])
        for i in range(len(csvFiles)): # limit nb of files for debugging, ex : csvFiles[:2]
            csvPath = csvFiles[i] 
            file_name = os.path.splitext(os.path.basename(csvPath))[0] # ex: A20E_seance2_0903_audio_farfield_trs
            file_name_new = file_name.split('_audio_farfield_trs')[0] # ex: A20E_seance2_0903
            print(file_name_new) # A20E_seance2_0903
            code = file_name_new.split('_')[0] # A20E
            seance = file_name_new.split('_')[1][6] #2
            logPaths = glob.glob(os.path.join(mainPath, f"{group}-subjects/{code}/{file_name_new}/Transcripts and Logs/{file_name_new}.log"))
            if logPaths:
                print(logPaths)
                info_EMO = getInfoFromMultimodal(code, seance)  # a dict of salientEmos
                data.append(read_from_dialogue_file_to_log_file(csvPath, logPaths, info_EMO))  # list of tuples
            else:
                print(f"Log file of '{file_name_new}' does not exist. Skipping...")
    df = create_dataframe(data)
    df.to_csv('User_Info_Stat.csv', index=False)


if __name__ == "__main__":
    main()
