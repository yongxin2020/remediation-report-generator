"""
To generate JSON file containing log and transcript info with the id of each subject
"""

def logDataClean(log_data): # 15:37:39:436 to 15:37:39
    import re
    # log_data = """
    # 15:37:39:436 - LOG|DIALOG|dialogue_choice
    # 15:37:39:486 - LOG|TXT|2352|** Choix du groupe
    # 15:37:40:603 - LOG|SETVAR|groupe|A
    # 15:37:40:636 - LOG|TXT|2693|** Choix Séance
    # """
    # Define a regular expression to capture only the hour and minutes and remove numerical identifiers
    pattern = re.compile(r'(\b\d{2}:\d{2}:\d{2}:\d{3}\s-\sLOG\|TXT\|)\d+\|(.*)')

    # Apply the regex for each line
    cleaned_log_data = '\n'.join([pattern.sub(r'\1\2', line) for line in log_data.split('\n')])

    # Define a regular expression to capture only the hour and minutes and remove milliseconds
    # 15:37:39:436 --> 15:37
    pattern2 = re.compile(r'(\b\d{2}:\d{2})\:\d{2}:\d{3} - ')

    # Apply the regex for each line
    cleaned_log_data2 = '\n'.join([pattern2.sub(r'\1 - ', line) for line in cleaned_log_data.split('\n')])

    return cleaned_log_data2

def readLogs(filename): # <class 'list'>, need to change into <class 'str'>
    global lines
    # type(lines[0]) # each line is a string
    if len(filename) == 1:
        # lines = open(filename[0], "r", encoding="utf-8").readlines()  # lines: <class 'list'>
        lines = open(filename[0], "r", encoding="utf-8").read()  # lines: <class 'string'>
        print(type(lines))
    elif len(filename) > 1: # for several sessions, there are more than one log file, need to combine them
        lines = []
        for i in range(len(filename)):
            lines_each_file = open(filename[i], "r", encoding="utf-8").readlines()  # lines: <class 'list'>
            lines.extend(lines_each_file)
    return lines

def convert_csv_columns_txt(csv_file):
    # reading csv file 
    text = open(csv_file, "r") 
      
    # joining with space content of text 
    text = ' '.join([i for i in text])   
      
    # replacing ',' by space 
    text = text.replace("\t", " ")   #","
    # print(text) #  K-Spch 01:21:12.585 4872.59 01:21:21.985 4881.98 00:00:09.400  9.40 c'était pas grave ça va c'était pas bon
    return text

def csv_dialog2txt(csv_file):
    # importing library 
    import pandas as pd 
      
    # Then loading csv file 
    df = pd.read_csv(csv_file, delimiter="\t", header=None) 
    # print(df)
    # converting the last column (dialogue) into list 
    # a = list(df.iloc[:,-1:]) 
    a = list(df[7]) 
    # Remove all the items of "None" in the dialogue csv
    a.remove('None')
      
    # converting list into string and then joining it with space 
    # b = ' '.join(str(e) for e in a if e != 'None') 
    # b = '\n'.join(map(str, a)) 
    b = '\n'.join(str(e) for e in a if e != 'None') 
      
    # printing result 
    return b

def shorten_dialog(dialog_str, max_token_length):
    utterances = dialog_str.split('\n')
    good_utterances = []
    total_length = 0
    for utterance in utterances:
        utterance_length = len(word_tokenize(utterance))
        if total_length + utterance_length > max_token_length:
            break
        total_length += utterance_length
        good_utterances.append(utterance)
    return '\n'.join(good_utterances)
