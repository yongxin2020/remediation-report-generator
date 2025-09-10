"""
Functions to extract session metadata and exercise results from log files.

# info_logs = getInfoFromLogs(filename)

Finally used in main_phrases_trous.py: session_date, end_time, results, propos_du_sujet
"""

import os
import re

def getInfoFromLogs(filename): # <class 'list'>, need to change into <class 'str'>
    print("filename in the getInfoFromLogs:", filename)
    global lines
    # type(lines[0]) # each line is a string
    # earlier version of 11/2022 may have one or two log files for one subject
    if len(filename) == 1:
        lines = open(filename[0], "r", encoding="utf-8").readlines()  # lines: <class 'list'>
    elif len(filename) > 1: # for several sessions, there are more than one log file, need to combine them
        lines = []
        for i in range(len(filename)):
            lines_each_file = open(filename[i], "r", encoding="utf-8").readlines()  # lines: <class 'list'>
            lines.extend(lines_each_file)

    results=[]
    affichage_avatar = []
    affichage_exercices = []
    répliques=[]
    propos_du_sujet=[]
    variables = []
    réponse_affect_start = ""
    rep_long_courte = ""
    start_time = ""
    end_time = ""
    session_date = ""

    for i in range(len(lines)):
        line = lines[i]
        if "ENRG" in line: # Début de l’enregistrement # 00:00:00:000 - ENRG|2022_05_04_11_24_14
            ENRG_date = re.split(' - ', line.strip('\n'))[1]
            session_date += ENRG_date
            print("session date:", session_date) # ENRG|2022_05_04_11_24_14
        if "STOP" in lines[-1]: # Fin de l’enregistrement # normal situation: one log file only # ex: A20E_seance1_0303 -> three log files
            end_t = re.split(' - ', lines[-1].strip('\n'))[0]
            end_time += end_t
        if "CONF|DIALOG" in line: # Affichage de l’avatar en plein écran
            avatar = re.split(' - ', line.strip('\n'))[0]
            affichage_avatar.append(avatar)
        if "CONF|EXERCICE" in line: # Affichage des exercices en plein écran
            exercices = re.split(' - ', line.strip('\n'))[0]
            affichage_exercices.append(exercices)
        if "LOG|TXT|" in line: # Réplique lue par Isabella
            réplique = line.strip('\n')
            répliques.append(réplique)
        if "LOG|REP|" and "Exo" in line: # Bouton cliqué par Isabella résumant les propos du sujet
            propos = line.strip('\n')
            propos = re.split("[|]| - ", propos)
            propos_du_sujet.append(propos[4].replace('**', '').replace(' Exo', 'Exo'))
            # print("propos_du_sujet:", propos_du_sujet) # propos_du_sujet: ['Exo 1 Retrouvez votre chemin', "Exo 7 Menez l'enquête"]
        if "LOG|ENDGAME|" in line: # Résultat de l’exercice : 3 valeurs possibles --> 1 : réussi, 0 : moyen, -1 : raté
            result=line.strip('\n')
            result = re.split("[|]| - ", result) # ['13:09:43:912', 'LOG', 'ENDGAME', '1'] --> results[3] : résultats
            results.append(result[3])
        if "LOG|SETVAR|" in line: # Enregistrement d’une variable, ex: LOG|SETVAR|exercice|
            variable=line.strip('\n')
            variables.append(variable)

        if "Comment allez-vous" in line: # not always "Comment allez-vous aujourd'hui ?", 
            # can also be "Comment vous sentez-vous aujourd'hui ?", or even simpler
            réponse = (lines[i + 1]).strip('\n')
            réponse = re.split("[|]", réponse)  # ['13:05:55:915 - LOG', 'REP', '1997', 'Bien réponse courte']
            réponse_extracted = réponse[-1]  # Bien réponse courte
            réponse_affect_start += (réponse_extracted.split())[0].lower()
            rep_long_courte += réponse_extracted.split()[2]
    
    del propos_du_sujet[::3] # propos_du_sujet list has 3*8 items, cause for each exo, appear three times

    # Finally used in main_phrases_trous.py: session_date, end_time, results, propos_du_sujet
    return {'session_date':session_date,
            'start_time':start_time,
            'end_time':end_time,
            'affichage_avatar':affichage_avatar,
            'affichage_exercices':affichage_exercices,
            'répliques':répliques,
            'propos_du_sujet':propos_du_sujet,
            'variables':variables,
            'results':results,
            'réponse_affect_start': réponse_affect_start, # extra info from log file
            'rep_long_courte': rep_long_courte}

# These are capabilities simulated by exercises used in the theradia corpus. You need to customise them for your own corpus.
def exo_stimulated_abilities(exe_order):
    if exe_order == 1 or exe_order == 2:
        exe_info = "Mémoire de travail visuelle, Mémoire visuo-spatiale"
    elif exe_order == 3 or exe_order == 4:
        exe_info = "Mémoire visuo-spatiale"
    elif exe_order == 5 or exe_order == 6:
        exe_info = "Langage, Logique grammaticale, Raisonnement"
    elif exe_order == 7 or exe_order == 8:
        exe_info = "Mémoire visuelle, Attention visuelle"
    elif exe_order == 9 or exe_order == 10:
        exe_info = "Manipulation des nombres, Traitement numérique"
    elif exe_order == 11 or exe_order == 12:
        exe_info = "Mémoire visuelle, Mémoire verbale"
    elif exe_order == 13 or exe_order == 14:
        exe_info = "Connaissance de la langue française"
    elif exe_order == 15 or exe_order == 16:
        exe_info = "Mémoire de travail, Planification, Raisonnement"
    else:
        print("No exercise information")
        exe_info = ""
    return exe_info

# Résultat de l’exercice : 3 valeurs possibles --> 1 : réussi, 0 : moyen, -1 : raté
def get_results_3_classes(results):
    ExoResultsDicts = {}
    exo_moyen_fonctions = exo_raté_fonctions = []
    exo_moyen_fonctions_str = exo_raté_fonctions_str = ""
    num_réussi = num_moyen = num_raté = 0
    for i in range(len(results)):
        result = results[i]
        exe_order = i+1
        exe_info = exo_stimulated_abilities(exe_order)
        ExoResultsDicts[i] = {"Num_exo": exe_order, "exe_info": exe_info,
                   "Exo_Result": result}

    for i in range(len(ExoResultsDicts)):
        print(ExoResultsDicts[i])
        if int(ExoResultsDicts[i]["Exo_Result"]) == -1:
            num_raté += 1
            exo_raté_fonctions.append(ExoResultsDicts[i]["exe_info"])
        if int(ExoResultsDicts[i]["Exo_Result"]) == 0:
            num_moyen += 1
            exo_moyen_fonctions.append(ExoResultsDicts[i]["exe_info"])
        if int(ExoResultsDicts[i]["Exo_Result"]) == 1:
            num_réussi += 1

    exo_raté_fonctions_set = list(set(exo_raté_fonctions))
    exo_moyen_fonctions_set = list(set(exo_moyen_fonctions))

    if len(exo_raté_fonctions_set) == 1:
        exo_raté_fonctions_str = exo_raté_fonctions_set[0]
    elif len(exo_raté_fonctions_set) > 1:
        exo_raté_fonctions_str += ", ".join(exo_raté_fonctions_set[:])

    if len(exo_moyen_fonctions_set) == 1:
        exo_moyen_fonctions_str = exo_moyen_fonctions_set[0]
    elif len(exo_moyen_fonctions_set) > 1:
        exo_moyen_fonctions_str += ", ".join(exo_moyen_fonctions_set[:])

    return ExoResultsDicts, num_réussi, num_moyen, num_raté, exo_moyen_fonctions_str, exo_raté_fonctions_str

def exo_score_to_str(score):
    if int(score) == -1:
        str = "❌" # raté or ❎
    elif int(score) == 0:
        str = "☑"
    elif int(score)==1:
        str = "✅"
    return str

def combine_exo_failed_twice(exo_failed_list): # , ExoTotalEssais
    # to provide a simplified version for exercises failed twice
    # format example: Exo 4 Jeux de blasons (7, 8/16)
    exo_failed_list_combined = []
    for i in range(len(exo_failed_list)):
        if i < len(exo_failed_list) - 1:
            if exo_failed_list[i].split("(")[0] == exo_failed_list[i+1].split("(")[0]:
                # old version: (1, 2/16), (11/16)
                # exoNum = int((exo_failed_list[i].split("(")[1]).split('/')[0]) # ex: 7, 8, etc
                # exo_failed_new = exo_failed_list[i].split("(")[0] + "({}, {}/{})".format(exoNum, exoNum + 1, ExoTotalEssais)
                # print(exo_failed_new) # Exo 4 Jeux de blasons (7, 8/16)
                # New: Change from (1, 2/16) to (1, 2e activités)
                exoNum = int((exo_failed_list[i].split("(")[1]).split("\u1D49")[0]) # ex: 7, 8, etc
                exo_failed_new = exo_failed_list[i].split("(")[0] + "({}, {}\u1D49 activités)".format(exoNum, exoNum + 1)
                exo_failed_list_combined.append(exo_failed_new)
            elif exo_failed_list[i].split("(")[0] != exo_failed_list[i-1].split("(")[0]:
                exo_failed_new = exo_failed_list[i]
                exo_failed_list_combined.append(exo_failed_new)
        elif i == len(exo_failed_list) - 1: # Attention, without this and if the last failed exo is different from the previous one, it will not be added.
            if exo_failed_list[i].split("(")[0] != exo_failed_list[i-1].split("(")[0]:
                exo_failed_new = exo_failed_list[i]
                exo_failed_list_combined.append(exo_failed_new)
    return exo_failed_list_combined

def exo_result_to_dict(propos_du_sujet, results):
    # using dictionary comprehension
    # to convert lists to dictionary
    exo_failed_list = []
    exo_failed = ""
    ExoTotalEssais = len(propos_du_sujet) # len(propos_du_sujet) should be 2*8; for MCI, 2*4
    for i in range(ExoTotalEssais):
        # propos_du_sujet[i] += " ({}/{})".format(i+1, ExoTotalEssais) # old version: (1, 2/16), (11/16)
        propos_du_sujet[i] += " ({}\u1D49 activité)".format(i + 1) # new version
    exo_res_list = {propos_du_sujet[i]: results[i] for i in range(ExoTotalEssais)}
    for key, value in exo_res_list.items():
        if int(value) == -1:
            exo_failed_list.append(key)
    # print(exo_failed_list)
    if len(exo_failed_list) == 0:
        exo_failed_phrase = ""
    elif len(exo_failed_list) == 1: # when there is only one activity failed
        exo_failed = exo_failed_list[0]
        exo_failed_phrase = f"L'exercice non réussi est : {exo_failed}."
    else:
        exo_failed_list_combined = combine_exo_failed_twice(exo_failed_list) #, ExoTotalEssais
        if len(exo_failed_list_combined) == 1:
            exo_failed = exo_failed_list_combined[0]
        elif len(exo_failed_list_combined) > 1:
            exo_failed += ", ".join(exo_failed_list_combined[:])
        # print(exo_failed_list_combined)
        # print(exo_failed)
        exo_failed_phrase = f"Les exercices non réussis sont : {exo_failed}."
    return exo_failed_phrase #exo_failed

def get_datetime_hour(date_time_str):
    from datetime import datetime
    # strptime(): Class method of datetime. Parse a string into a datetime object given a corresponding format
    date_time_obj = datetime.strptime(date_time_str, '%H:%M:%S:%f')
    datetime_hour = date_time_obj.hour
    return datetime_hour