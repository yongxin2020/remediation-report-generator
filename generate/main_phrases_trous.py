"""
Definition of the useful patterns of cloze sentences for the generation.

Main file of the generation system
Deal with log / dialogue file input using getInfoFromLogs() / getInfoFromDialogue() functions
Use EmotionAnalysisPipeline() to obtain predictions and analysis results from multimodal emotion recognition
Info used in the final version:
    - From log file: session_date, end_time, results, propos_du_sujet
    - From csv file (dialogue transcript): num_tokens_unique, lexical_diversity_TTR, duration_utterance, num_output_pho, num_token_per_utterance
Use toPreformattedText() for the generation of reports

    example: python ./main_phrases_trous.py "code:A15E;seance:2"
"""

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import json
import getopt
import random
import statistics
from mdutils.mdutils import MdUtils
from mdutils import Html
from dialogues_theradia import *
from logs_theradia import *
from calculate_norme import get_quartile_range, compare_value_to_norm
import yaml
from pathlib import Path
import os
sys.path.append(os.path.abspath(os.path.join('..', 'emotion_analysis')))
from emotion_pipeline import EmotionAnalysisPipeline

path_to_yaml = "../settings.yaml"

with open(path_to_yaml, 'r') as file:
    config = yaml.safe_load(file)

# Informations contextuelles
P1 = "La séance du {} a eu lieu vers {}. Durant cette séance, le patient a réalisé {} activités ({} exercices réalisés deux fois) pendant {}. Le Tableau 1 résume les fonctions cognitives travaillées ainsi que le résultat des activités."
P2_1_num1 = "Parmi ces activités : {} activité n'a pas été réussie (taux de bonnes réponses < 60%)." # sollicitant {}
P2_1_num0 = "Parmi ces activités : " # "Parmi ces activités : {} activité n'a pas été réussie (taux de bonnes réponses < 60%)." # sollicitant {}
P2_1_numOthers = "Parmi ces activités : {} activités n'ont pas été réussies (taux de bonnes réponses < 60%)." # sollicitant {}

# Results
P2_2_num1 = "{} activité a été partiellement réussie (taux de bonnes réponses entre 60% et 80%)."
P2_2_num0 = "{} activité a été partiellement réussie (taux de bonnes réponses entre 60% et 80%)."
P2_2_numOthers = "{} activités ont été partiellement réussies (taux de bonnes réponses entre 60% et 80%)."
P2_3 = "Les autres activités montrent des résultats tout à fait satisfaisants (taux de bonnes réponses > 80%)."

P3 = "Le taux de réussite des exercices est de {}%."
P4 = "{}" # Les exercices non réussis sont : 

# Affect
Affec_P1 = "Durant la séance, le patient s'est montré particulièrement {}, mais aussi {} par rapport aux émotions ressenties par les patients du même groupe." # "mais aussi", seperated emo positives and negatives
Affec_P2 = "Durant la séance, le patient s'est montré particulièrement {} par rapport aux émotions ressenties par les patients du même groupe." # only positive or negative emotions

# Language
Langage_comparaison_P0 = "Le Tableau 2 ci-dessous donne les valeurs des indicateurs linguistiques calculés sur les énoncés du patient lors de l'interaction. Les explications des différents indicateurs sont fournies en Annexe."
Langage_comparaison_P1 = "Par rapport à la norme, {}" #plus élevée
Langage_comparaison_P2 = "Au contraire, {}."

def toPreformattedText(info_logs, info_dialogue, info_EMO, id_seance_num):
    text = ""

    # P1
    session_date = info_logs['session_date'] # ENRG|2022_05_04_11_24_14
    month = session_date[10:12]
    day = session_date[13:15]
    month_dict = {"01": "Janvier", "02": "Février", "03": "Mars", "04": "Avril", "05": "Mai", "06": "Juin", "07": "Juillet", "08": "Août", "09": "Septembre", "10": "Octobre", "11": "Novembre", "12": "Décembre"}
    month_in_string = month_dict.get(month)
    date_session_string = day + " " + month_in_string
    # Split the session date string using underscores
    date_components = session_date.split('_')
    # Extract the time components (hours, minutes, seconds)
    start_time_hour = int(date_components[3])
    start_time_minute = int(date_components[4])
    # Format the time components into "HH:MM" format
    start_time = f"{start_time_hour}:{start_time_minute}"
    print("Start Time:", start_time)
    if 0 <= start_time_minute < 15:
        textual_start_time = str(start_time_hour) + " heures"
    elif 15 <= start_time_minute < 45:
        textual_start_time = str(start_time_hour) + " heures et demie" # 11 heures et demie
    elif 45 <= start_time_minute < 60:
        textual_start_time = str(start_time_hour + 1) + " heures"

    end_time = info_logs['end_time'] # 13:51:22:157 => in the new log version: 01:17:49:249
    end_time_hour = int(end_time[:2])
    end_time_minute = int(end_time[3:5])

    duration_session = end_time_hour*60 + end_time_minute
    print("duration session:", duration_session)

    if duration_session < 30:
        duration_session_str = "moins d'une demi-heure"
    elif 30 <= duration_session < 45:
        duration_session_str = "environ une demi-heure"
    elif 45 <= duration_session < 75:
        duration_session_str = "environ une heure"
    elif 75 <= duration_session < 90:
        duration_session_str = "environ une heure et demie"
    elif duration_session >= 90:
        duration_session_str = "plus d'une heure et demie"
        
    nb_exercices = int(len(info_logs['results']) / 2)
    nb_activités = nb_exercices*2

    # P2
    results = info_logs['results']
    ExoResultsDicts, num_réussi, num_moyen, num_raté, exo_moyen_fonctions, exo_raté_fonctions = get_results_3_classes(results)

    # P3
    taux_réussite = round(num_réussi / len(info_logs['results'])*100, 1)
    
    # P4
    propos_du_sujet = info_logs['propos_du_sujet']
    exo_failed_phrase = exo_result_to_dict(propos_du_sujet, results) # exo_failed_phrase: a phrase contains three different conditions: non failed exo, 1 failed exo, >1 exo

    # Affec_P1
    if info_EMO == {}: # emotion recognition predictions have only be generated for M subjects
        states = ["actif", "passif", "agité"]
        salientEmotions = random.choice(states)
        salientEmotions_Negative = salientEmotions_Positive = ""
        salientEmo_Negative_list = salientEmo_Positive_list = []
    else:
        # info_EMO: a dict like: {'satisfied': 10, 'confident': 9, 'anxious': 6, 'relaxed': 5, 'annoyed': 3, 'happy': 2, 'interested': 2, 'desperate': 1}
        # Calculate the median value
        values = list(info_EMO.values())
        median_value = sorted(values)[len(values) // 2]

        # Report emotions above the median
        emotions_above_median = {emotion: value for emotion, value in info_EMO.items() if value > median_value}
        print("Emotions above the median:")
        for emotion, value in emotions_above_median.items():
            print(f"{emotion}: {value}")

        positive_emotions = ['détendu', 'intéressé', 'satisfait', 'confiant', 'heureux']
        negative_emotions = ['frustré', 'surpris', 'ennuyé', 'désespéré', 'anxieux']

        # Mapping dictionary for translation
        translation_mapping = {
            'relaxed': 'détendu',
            'interested': 'intéressé',
            'satisfied': 'satisfait',
            'confident': 'confiant',
            'happy': 'heureux',
            'frustrated': 'frustré',
            'surprised': 'surpris',
            'annoyed': 'ennuyé',
            'desperate': 'désespéré', 
            'anxious': 'anxieux'
        }
        # Create a new dictionary with translated keys
        emotions_above_median_translated = {translation_mapping.get(key, key): value for key, value in emotions_above_median.items()}
        salientEmotions = ', '.join(emotions_above_median_translated.keys())

        # Separate emotions into positive and negative categories
        salientEmo_Positive = {key: value for key, value in emotions_above_median_translated.items() if
                               key in positive_emotions}
        salientEmo_Negative = {key: value for key, value in emotions_above_median_translated.items() if
                               key in negative_emotions}
        
        salientEmo_Negative_list = list(salientEmo_Negative.keys())
        salientEmo_Positive_list = list(salientEmo_Positive.keys())
        
        # Convert positive and negative emotions into comma-separated strings
        salientEmotions_Positive = ', '.join(salientEmo_Positive.keys())
        salientEmotions_Negative = ', '.join(salientEmo_Negative.keys())
        print(salientEmotions_Positive)
        print(salientEmotions_Negative)

    # Langage_P1
    num_tokens_unique = info_dialogue['num_tokens_unique']
    lexical_diversity_TTR = info_dialogue['lexical_diversity_TTR']
    lexical_diversity_BI = info_dialogue['lexical_diversity_BI']
    lexical_diversity_HS = info_dialogue['lexical_diversity_HS']
    c_density_score = info_dialogue['c_density_score']
    p_density_score = info_dialogue['p_density_score']

    # Langage_P2
    # sum_duration_speak, débit_parole
    duration_utterance = info_dialogue['duration_utterance'] # 'duration_utterance' is a list
    sum_duration_speak = sum(duration_utterance) # in seconds
    print("duration_utterance:", duration_utterance)
    print("sum duration speak in seconds:", sum_duration_speak) # 1133.1 secondes
    # Temps moyen de parole par heure (minutes) = round(((sum_duration_speak/60)/duration_session)*60, 1) # => Temps moyen de parole par heure (minutes)
    # => the variable for comparaison is not sum_duration_speak, but sum_duration_speak/duration_session, which is sec/mins, and also equals to mins/hour => Temps moyen de parole par heure (minutes)
    print("Temps moyen de parole par heure (minutes) :", sum_duration_speak/duration_session)
    # count speech rate by phonemes/secs
    num_output_pho = info_dialogue['num_output_pho']
    débit_parole = round(sum(num_output_pho) / sum_duration_speak, 1) # (phonèmes/sec)

    # Langage_P3
    num_token_per_utterance = info_dialogue['num_token_per_utterance'] # Taille moyenne d’énoncé (mots)
    num_token_per_utterance_mean = round(sum(num_token_per_utterance) / len(num_token_per_utterance),
                                         1)  # round(average, 2)
    num_token_per_utterance_max = max(num_token_per_utterance)
    num_token_per_utterance_min = min(num_token_per_utterance)
    num_token_per_utterance_ecart_type = round(statistics.pstdev(num_token_per_utterance), 1)

    # Langage_P4
    duration_utterance_mean = round(sum(duration_utterance) / len(duration_utterance), 1) # Durée moyenne d’énoncé (sec)
    duration_utterance_max = max(duration_utterance)
    duration_utterance_min = min(duration_utterance)
    duration_utterance_ecart_type = round(statistics.pstdev(duration_utterance), 1)

    # write variables to text
    text +="<p>"
    text += P1.format(date_session_string, textual_start_time, nb_activités, nb_exercices, duration_session_str) + " "
    text += "<ul>"
    text += "<li>"
    if num_raté == 1:
        text += P2_1_num1.format(num_raté) + " "
    elif num_raté == 0:
        text += P2_1_num0 + " "
    else:
        text += P2_1_numOthers.format(num_raté) + " "
    text += "</li>"
    text += "<li>"
    # P2_2
    if num_moyen == 1:
        text += P2_2_num1.format(num_moyen) + " "
    elif num_moyen == 0:
        text += P2_2_num0.format("Aucune") + " "
    else:
        text += P2_2_numOthers.format(num_moyen) + " "
    text += "<li>"
    text += P2_3.format() + " "
    text += "</li>"
    text += "</ul>"
    text += P3.format(taux_réussite) + " "
    text += P4.format(exo_failed_phrase) + " \n"
    text +="</p>"

    text +="<p>"
    if len(salientEmo_Negative_list) == 0 and len(salientEmo_Positive_list) == 0:
        text = ""
    elif len(salientEmo_Negative_list) == 0:
        text += Affec_P2.format(salientEmotions_Positive) + " \n"
    elif len(salientEmo_Positive_list) == 0:
        text += Affec_P2.format(salientEmotions_Negative) + " \n"
    elif len(salientEmo_Negative_list) > len(salientEmo_Positive_list):
        text += Affec_P1.format(salientEmotions_Negative, salientEmotions_Positive) + " \n"
    elif len(salientEmo_Negative_list) < len(salientEmo_Positive_list):
        text += Affec_P1.format(salientEmotions_Positive, salientEmotions_Negative) + " \n"
    else:
        text = ""
    text +="</p>"
    
    # because of the need of intergrating tables and images into the text, compose text by paragraph
    ParagraphContextualInfo = ""
    ParagraphResults = ""
    ParagraphAffect = ""
    ParagraphLanguageComparaison = ""

    ParagraphContextualInfo += "<p>"
    ParagraphContextualInfo += P1.format(date_session_string, textual_start_time, nb_activités, nb_exercices, duration_session_str) + " "

    ParagraphResults += "<p>"
    # P2_1
    if num_raté == 1:
        ParagraphResults += P2_1_num1.format(num_raté) + " "
    elif num_raté == 0:
        ParagraphResults += P2_1_num0 + " "
    else:
        ParagraphResults += P2_1_numOthers.format(num_raté) + " "
    # P2_2
    if num_moyen == 1:
        ParagraphResults += P2_2_num1.format(num_moyen) + " "
    elif num_moyen == 0:
        ParagraphResults += P2_2_num0.format("Aucune") + " "
    else:
        ParagraphResults += P2_2_numOthers.format(num_moyen) + " " # exo_moyen_fonctions
    ParagraphResults += P2_3.format() + " "
    ParagraphResults += "</p>"
    ParagraphResults += P3.format(taux_réussite) + " "
    ParagraphResults += P4.format(exo_failed_phrase) + " \n"
    ParagraphResults += "</p>"

    ParagraphAffect +="<p>"
    if len(salientEmo_Negative_list) == 0 and len(salientEmo_Positive_list) == 0:
        ParagraphAffect = ""
    elif len(salientEmo_Negative_list) == 0:
        ParagraphAffect += Affec_P2.format(salientEmotions_Positive) + " \n"
    elif len(salientEmo_Positive_list) == 0:
        ParagraphAffect += Affec_P2.format(salientEmotions_Negative) + " \n"
    elif len(salientEmo_Negative_list) > len(salientEmo_Positive_list):
        ParagraphAffect += Affec_P1.format(salientEmotions_Negative, salientEmotions_Positive) + " \n"
    elif len(salientEmo_Negative_list) < len(salientEmo_Positive_list):
        ParagraphAffect += Affec_P1.format(salientEmotions_Positive, salientEmotions_Negative) + " \n"
    else:
        ParagraphAffect = ""
    ParagraphAffect +="</p>"

    # Create Tuples, which are immutable
    User_Info = (date_session_string, textual_start_time, nb_activités, nb_exercices, duration_session_str,
                 start_time, duration_session,
                 num_réussi, num_moyen, num_raté,
                 taux_réussite,
                 num_tokens_unique,
                 sum_duration_speak, débit_parole,
                 num_token_per_utterance_mean, num_token_per_utterance_max, num_token_per_utterance_min,
                 duration_utterance_mean, duration_utterance_max, duration_utterance_min,
                 lexical_diversity_TTR, lexical_diversity_BI, lexical_diversity_HS,
                 c_density_score, p_density_score)

    # Prepare the dictionary information to build the table
    TableDict = {}
    RowDicts = {}
    RowDicts[0] = {"Indicateur": "Taille du vocabulaire", "Valeur": num_tokens_unique,
                   "Comparaison": compare_value_to_norm(num_tokens_unique,
                                                       get_quartile_range("num_tokens_unique", 0)),
                   "Norme": get_quartile_range("num_tokens_unique", 0)}
    # Normalise speaking time relative to session duration: get_quartile_range("sum_duration_speak_per_minute"); seconds/minute = minutes/hour
    RowDicts[1] = {"Indicateur": "Temps moyen de parole par heure", "Valeur": round(sum_duration_speak/duration_session),
                   "Comparaison": compare_value_to_norm(round(sum_duration_speak/duration_session),
                                                       get_quartile_range("sum_duration_speak_per_minute", 0)),
                   "Norme": get_quartile_range("sum_duration_speak_per_minute", 0)}
    RowDicts[2] = {"Indicateur": "Débit de parole", "Valeur": débit_parole,
                   "Comparaison": compare_value_to_norm(débit_parole, get_quartile_range("débit_parole", 1)),
                   "Norme": get_quartile_range("débit_parole", 1)}
    RowDicts[3] = {"Indicateur": "Taille moyenne d'énoncé", "Valeur": f"{num_token_per_utterance_mean} ± {num_token_per_utterance_ecart_type}",
                   "Comparaison": compare_value_to_norm(num_token_per_utterance_mean, get_quartile_range(
                       "num_token_per_utterance_mean", 1)),
                   "Norme": get_quartile_range("num_token_per_utterance_mean", 1)}
    RowDicts[4] = {"Indicateur": "Durée moyenne d'énoncé", "Valeur": f"{duration_utterance_mean} ± {duration_utterance_ecart_type}",
                   "Comparaison": compare_value_to_norm(duration_utterance_mean,
                                                       get_quartile_range("duration_utterance_mean", 1)),
                   "Norme": get_quartile_range("duration_utterance_mean", 1)}
    RowDicts[5] = {"Indicateur": "Diversité lexicale", "Valeur": lexical_diversity_TTR,
                   "Comparaison": compare_value_to_norm(lexical_diversity_TTR,
                                                       get_quartile_range("lexical_diversity_TTR", 2)),
                   "Norme": get_quartile_range("lexical_diversity_TTR", 2)}
    RowDicts[6] = {"Indicateur": "Densité lexicale du contenu", "Valeur": c_density_score,
                    "Comparaison": compare_value_to_norm(c_density_score,
                                                        get_quartile_range("c_density_score", 2)),
                    "Norme": get_quartile_range("c_density_score", 2)}

    for i in range(7):
        TableDict[f"Row_{i}"] = RowDicts[i]

    list_indicateur_higher = []
    list_indicateur_lower = []
    indicateur_higher = ""
    indicateur_lower = ""
    for i in range(len(TableDict)):
        if TableDict[f"Row_{i}"]["Comparaison"] == "&#129045;":
            nom_indicateur_high = TableDict[f"Row_{i}"]["Indicateur"]
            list_indicateur_higher.append(nom_indicateur_high.lower())
        elif TableDict[f"Row_{i}"]["Comparaison"] == "&#129047;":
            nom_indicateur_low = TableDict[f"Row_{i}"]["Indicateur"]
            list_indicateur_lower.append(nom_indicateur_low.lower())

    # Langage_comparaison_P1
    if len(list_indicateur_higher) == 1:
        indicateur_higher = list_indicateur_higher[0]
        indicateur_higher = f"la valeur de \"{indicateur_higher}\" est plus élevée."
    elif len(list_indicateur_higher) > 1:
        indicateur_higher += ", ".join(list_indicateur_higher[:])
        indicateur_higher = f"les valeurs de \"{indicateur_higher}\" sont plus élevées."

    # Langage_comparaison_P2
    if len(list_indicateur_lower) == 1:
        indicateur_lower = list_indicateur_lower[0]
        indicateur_lower = f"la valeur de \"{indicateur_lower}\" est plus faible"
    elif len(list_indicateur_lower) > 1:
        indicateur_lower += ", ".join(list_indicateur_lower[:])
        indicateur_lower = f"les valeurs de \"{indicateur_lower}\" sont plus faibles"

    ParagraphLanguageComparaison += "<p>"
    ParagraphLanguageComparaison += Langage_comparaison_P0 + " "
    if len(list_indicateur_higher) != 0:
        ParagraphLanguageComparaison += Langage_comparaison_P1.format(indicateur_higher) + " "
    else:
        pass
    if len(list_indicateur_lower) != 0:
        ParagraphLanguageComparaison += Langage_comparaison_P2.format(indicateur_lower) + " \n"
    else:
        pass
    ParagraphLanguageComparaison += "</p>"

    # Create a dictionary of the different exercises with the corresponding comments, for later creation of the table.
    Exo_results_TableDict = {}
    Exo_results_RowDicts = {}

    if len(results) == 16:
        Exo_results_RowDicts[0] = {"Exercice": "Retrouvez votre chemin",
                                   "Capacités stimulées": "Mémoire de travail visuelle, Mémoire visuo-spatiale",
                                   "N1": exo_score_to_str(results[0]), "N2": exo_score_to_str(results[1])}
        Exo_results_RowDicts[1] = {"Exercice": "Objets, où êtes-vous ?",
                                   "Capacités stimulées": "Mémoire visuo-spatiale",
                                   "N1": exo_score_to_str(results[2]), "N2": exo_score_to_str(results[3])}
        Exo_results_RowDicts[2] = {"Exercice": "Que d'accrocs dans cette histoire",
                                   "Capacités stimulées": "Langage, Logique grammaticale, Raisonnement",
                                   "N1": exo_score_to_str(results[4]), "N2": exo_score_to_str(results[5])}
        Exo_results_RowDicts[3] = {"Exercice": "Jeux de blasons",
                                   "Capacités stimulées": "Mémoire visuelle, Attention visuelle",
                                   "N1": exo_score_to_str(results[6]), "N2": exo_score_to_str(results[7])}
        Exo_results_RowDicts[4] = {"Exercice": "Mettez de l'ordre dans ces comptes",
                                   "Capacités stimulées": "Manipulation des nombres, Traitement numérique",
                                   "N1": exo_score_to_str(results[8]), "N2": exo_score_to_str(results[9])}
        Exo_results_RowDicts[5] = {"Exercice": "Garçon SVP !",
                                   "Capacités stimulées": "Mémoire visuelle, Mémoire verbale",
                                   "N1": exo_score_to_str(results[10]), "N2": exo_score_to_str(results[11])}
        Exo_results_RowDicts[6] = {"Exercice": "Menez l'enquête",
                                   "Capacités stimulées": "Connaissance de la langue française",
                                   "N1": exo_score_to_str(results[12]), "N2": exo_score_to_str(results[13])}
        Exo_results_RowDicts[7] = {"Exercice": "Tours de Hanoi",
                                   "Capacités stimulées": "Mémoire de travail, Planification, Raisonnement",
                                   "N1": exo_score_to_str(results[14]), "N2": exo_score_to_str(results[15])}

        for i in range(8):
            Exo_results_TableDict[f"Row_{i}"] = Exo_results_RowDicts[i]

    elif len(results) == 8:
        Exo_results_RowDicts[0] = {"Exercice": "Retrouvez votre chemin",
                                   "Capacités stimulées": "Mémoire de travail visuelle, Mémoire visuo-spatiale",
                                   "N1": exo_score_to_str(results[0]), "N2": exo_score_to_str(results[1])}
        Exo_results_RowDicts[1] = {"Exercice": "Objets, où êtes-vous ?",
                                   "Capacités stimulées": "Mémoire visuo-spatiale",
                                   "N1": exo_score_to_str(results[2]), "N2": exo_score_to_str(results[3])}
        Exo_results_RowDicts[2] = {"Exercice": "Que d'accrocs dans cette histoire",
                                   "Capacités stimulées": "Langage, Logique grammaticale, Raisonnement",
                                   "N1": exo_score_to_str(results[4]), "N2": exo_score_to_str(results[5])}
        Exo_results_RowDicts[3] = {"Exercice": "Menez l'enquête",
                                   "Capacités stimulées": "Connaissance de la langue française",
                                   "N1": exo_score_to_str(results[6]), "N2": exo_score_to_str(results[7])}

        for i in range(4):
            Exo_results_TableDict[f"Row_{i}"] = Exo_results_RowDicts[i]

    # create a JSON files storing variables values at the same time
    dictionary = {
        "date_session_string": date_session_string,
        "textual_start_time": textual_start_time,
        "nb_activités": nb_activités,
        "nb_exercices": nb_exercices,
        "duration_session_str": duration_session_str,
        "num_raté": num_raté,
        "num_moyen": num_moyen,
        "taux_réussite": taux_réussite,
        "exo_failed": exo_failed_phrase,
        "salientEmotions": salientEmotions,
        "Exo_results_TableDict": Exo_results_TableDict,
        "TableDict": TableDict
    }
    print(dictionary)
    # Serializing json
    json_object = json.dumps(dictionary, indent=4, ensure_ascii=False)
     
    # Writing to sample.json
    with open(f"../MRG_LLMs/{id_seance_num}.json", "w") as outfile:
        outfile.write(json_object)

    return text, User_Info, TableDict, Exo_results_TableDict, ParagraphContextualInfo, ParagraphResults, ParagraphAffect, ParagraphLanguageComparaison # ParagraphDifficulty, ParagraphCommunication, ParagraphTendance


def make_markdown_file(ParagraphContextualInfo, ParagraphResults, ParagraphAffect, ParagraphLanguageComparaison, md_file_name, TableDict, Exo_results_TableDict): # ParagraphDifficulty, ParagraphCommunication, ParagraphTendance
    mdFile = MdUtils(file_name='Report_MD_files/report_{}'.format(md_file_name))

    mdFile.new_header(level=1, title='Rapport de séance')  # style is set 'atx' format by default.

    mdFile.new_paragraph(ParagraphContextualInfo)
    mdFile.new_paragraph(ParagraphResults)
    mdFile.new_line()

    # Inline Table: Table on performance of exercises
    mdFile.new_header(level=2, title='Tableau 1 : Exercices et fonctions cognitives traitées')
    list_of_strings = ["Exercice", "Capacités stimulées", "Essais 1", "Essais 2"]
    rows = list(Exo_results_TableDict.keys())
    # len(Exo_results_TableDict): number of exercices, MCI 4 exercices, A/J 8 exercices
    num_rows = len(Exo_results_TableDict) + 1
    for i in range(len(Exo_results_TableDict)):
        list_of_strings.extend(
            [Exo_results_TableDict[rows[i]]["Exercice"], Exo_results_TableDict[rows[i]]["Capacités stimulées"],
             Exo_results_TableDict[rows[i]]["N1"], Exo_results_TableDict[rows[i]]["N2"]])
    mdFile.new_line()
    mdFile.new_table(columns=4, rows=num_rows, text=list_of_strings, text_align='center')
    mdFile.new_line("Seuils : ", bold_italics_code='b')
    mdFile.new_line("✅ réussi = exactitude > 80%")
    mdFile.new_line("☑ moyen  = exactitude entre 60% et 80%")
    mdFile.new_line("❌ non réussi = exactitude < 60%")

    mdFile.new_paragraph(ParagraphAffect)

    mdFile.new_paragraph(ParagraphLanguageComparaison)
    mdFile.new_line()

    # Inline Table
    mdFile.new_header(level=2, title='Tableau 2 : Indicateurs linguistiques')
    list_of_strings = ["Indicateur", "Valeur", "Comparaison", "Norme"]
    rows = list(TableDict.keys())
    for i in range(len(TableDict)):
        list_of_strings.extend([TableDict[rows[i]]["Indicateur"], TableDict[rows[i]]["Valeur"], TableDict[rows[i]]["Comparaison"], TableDict[rows[i]]["Norme"]])
    mdFile.new_line()
    mdFile.new_table(columns=4, rows=8, text=list_of_strings, text_align='center')
    mdFile.write("**Note sur la norme** : 37 séances de 20 participants d'un groupe de personnes de la même tranche d'âge que le patient (en utilisant la médiane et les 1\u1D49 et 3\u1D49 quartiles).") # previously 39 sessions

    mdFile.new_line()

    mdFile.new_header(level=2, title='Annexe : Explications des indicateurs linguistiques')

    # Add explanations for the indicators
    mdFile.write("- **Taille du vocabulaire** : nombre de mots uniques.")
    mdFile.write('  \n')
    mdFile.write("- **Temps moyen de parole par heure** : en minutes.")
    mdFile.write('  \n')
    mdFile.write("- **Débit de parole** : nombre de phonèmes par unité de temps en seconde.")
    mdFile.write('  \n')
    mdFile.write("- **Taille moyenne d'énoncé** : nombre moyen de mots par énoncé.")
    mdFile.write('  \n')
    mdFile.write("- **Durée moyenne d'énoncé** : en secondes.")
    mdFile.write('  \n')
    mdFile.write("- **Diversité lexicale** : nombre de mots uniques divisé par le nombre total de mots.")
    mdFile.write('  \n')
    mdFile.write("- **Densité lexicale du contenu** : nombre de contenus exprimés (verbes, noms, adjectifs, adverbes) divisé par le nombre total de mots.") #, align='center'

    mdFile.create_md_file()

# Create a folder to save generated reports in .md format
def createMDDirectoryStructure():
    md_directory = 'Report_MD_files'
    if not os.path.exists(md_directory):
        os.makedirs(md_directory)

def read_from_dialogue_file_to_log_file(dialogue_file, log_file, info_EMO):
    id_seance_num = os.path.splitext(os.path.basename(log_file[0]))[0] # `log_file` is a list -> A01A_seance1_1512
    print("id_seance_num:", id_seance_num)
    user_id = id_seance_num.split('_')[0]  # user_id: A01A
    print("user_id:", user_id)

    age_group = user_id[0] # A
    induction_group = user_id[3] # E
    info_dialogue = getInfoFromDialogue(dialogue_file)
    info_logs = getInfoFromLogs(log_file)
    text, User_Info, TableDict, Exo_results_TableDict, ParagraphContextualInfo, ParagraphResults, ParagraphAffect, ParagraphLanguageComparaison = toPreformattedText(info_logs, info_dialogue, info_EMO, id_seance_num)

    # Comment the following line out when doing statistics
    make_markdown_file(ParagraphContextualInfo, ParagraphResults, ParagraphAffect, ParagraphLanguageComparaison, id_seance_num, TableDict, Exo_results_TableDict)

    # Duplicate tuples and add new items
    User_Info_update = (user_id, id_seance_num, age_group, induction_group) + User_Info # Concatenate multiple tuples
    print(text)
    return User_Info_update

def generate(code: str, seance: int) -> None:
    """Generate files using paths and patterns from YAML."""
    # Get base paths from config
    main_path = Path(config["paths"]["main"])
    transcripts_dir = config["paths"]["transcripts"]  # "Transcripts and Logs"

    createMDDirectoryStructure()

    # Determine subject group (e.g., "M" → "M-subjects")
    group_prefix = config["subjects"]["group_prefixes"].get(code[0], "")
    if not group_prefix:
        raise ValueError(f"No group prefix defined for code: {code}")

    # Construct paths using dynamic placeholders
    session_dir = main_path / group_prefix / code / f"{code}_seance{seance}_*"
    transcripts_path = session_dir / transcripts_dir

    print("transcripts_path", transcripts_path)

    # Get file patterns from YAML
    csv_pattern = config["file_patterns"]["dialogue_csv"].format(participant=code, num_seance=seance)
    log_pattern = config["file_patterns"]["log_file"].format(participant=code, num_seance=seance)

    # Find files (with error handling)
    dialogue_files = glob.glob(str(transcripts_path / csv_pattern))
    log_files = glob.glob(str(transcripts_path / log_pattern))

    if not dialogue_files:
        raise FileNotFoundError(f"No dialogue files found matching: {csv_pattern}")
    if not log_files:
        raise FileNotFoundError(f"No log files found matching: {log_pattern}")

    print("Dialogue files:", dialogue_files)
    print("Log files:", log_files)
    
    # Initialize emotion pipeline
    emotion_pipeline = EmotionAnalysisPipeline(
        settings_path="../settings.yaml",
        participant=code,
        session=seance
    )

    # Run complete analysis (replaces getInfoFromMultimodal)
    emotion_results = emotion_pipeline.run_analysis(
        audio_source=str(transcripts_path),
        video_source=str(transcripts_path)
    )

    print("emotion_results:", emotion_results)

    # Extract salient emotions in expected format
    info_EMO = {}
    if emotion_results['salient_emotions']:
        info_EMO = {
            emo: intensity
            for emo, intensity in emotion_results['salient_emotions'].items()
        }

    # Process file with emotion results
    read_from_dialogue_file_to_log_file(
        dialogue_files[0], 
        log_files, 
        info_EMO
    )


def usage():
    """
        affiche l'aide
    """
    print("Usage:\n " + sys.argv[0] + " code participant séance")
    print("\n\texemple:\t " + sys.argv[0] + ' "code:A15E;seance:2"')
    print("\n\t\t\t\t -> génère le résumé de la séance 2 pour le patient A15E")


def main():
    """
    fonction principale du programme de génération.
    """
    if sys.argv is None or len(sys.argv) < 2:
        usage()
        exit()

    niveau = []
    valeur = []
    for pair in sys.argv[1].split(';'):
        sep = pair.find(':')
        if sep < 1:
            print("Error: valeur de code et seance mal formées")
            exit()
        niveau.append(pair[0:sep])
        valeur.append(pair[sep + 1:])

    generate(valeur[0], valeur[1])
    
    os.system('pandoc -s Report_MD_files/report_{}_seance{}_????.md --metadata pagetitle="{}" -o report_{}_seance{}.html'.format(valeur[0], valeur[1], "Exemple participant_{}".format(valeur[0]), valeur[0],valeur[1]))

    
if __name__ == "__main__":
    main()


