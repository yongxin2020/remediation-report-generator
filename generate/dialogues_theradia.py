# -*- coding: utf-8 -*-

"""
Functions to extract information from dialogue transcripts (CSV).

# info_dialogue = getInfoFromDialogue(filename)

In the final version, only the following elements have been used:
num_tokens_unique, duration_utterance, num_token_per_utterance, num_output_pho, lexical_diversity_TTR
"""

import csv
import glob
import os
import re
import pandas as pd
import numpy as np
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import epitran


def remove_punc_esp_2_phonemes(text):
    # Library for text-to-phonemes
    epi = epitran.Epitran('fra-Latn-p')  # in fra-Latn-p script
    # Removing punctuations in string
    # Using regex
    res = re.sub(r'[^\w\s]', '', text)
    text_for_pho = re.sub(r' ', '', res)

    output_pho = epi.transliterate(text_for_pho)
    num_output_pho = len(output_pho)

    return output_pho, num_output_pho

def readCSVFile(filename):
    # reader = pd.read_csv(filename, delimiter = "\t", header=None)
    reader = pd.read_csv(filename, header=0)  # Specify that the first row contains column names
    return reader

def get_dialogue_content(df_dialogue_no_none):
    dialogue = ""
    dialogue_lines = df_dialogue_no_none["content"]
    for line in dialogue_lines:
        dialogue += line + "\n"
    return dialogue

# Detect questions (asked in French)
def find_questions(df_dialogue_no_none):
    dialogue_lines = df_dialogue_no_none["content"]
    questions = []
    question_words = ["est-ce que", "est-ce qui", "comment", "pourquoi"]

    for line in dialogue_lines:
        if any(x in line.lower() for x in question_words):
            questions.append(line)
    return questions

def get_Speech_Info(df_dialogue_no_none):
    duration_utterance = []
    num_token_per_utterance = []  # taille moyenne d'énoncé
    list_utterance = []
    speech_rate_per_utterance = []
    num_output_pho = []
    output_pho = []
    for ind in df_dialogue_no_none.index: # The index labels of the DataFrame
        # duration of each utterance
        # need to convert a measurement in milliseconds to a measurement in seconds
        durée = round(df_dialogue_no_none["end_time"][ind]/1000 - df_dialogue_no_none["start_time"][ind]/1000, 3) # utterance duration in seconds
        duration_utterance.append(durée)
        # taille moyenne de chaque énoncé (tokens)
        utterance = df_dialogue_no_none["content"][ind]
        list_utterance.append(utterance)
        token_per_utterance = word_tokenize(utterance, language="french") 
        num_token = len(token_per_utterance)
        num_token_per_utterance.append(num_token)
        speech_rate_each = float(num_token) / float(durée)
        speech_rate_per_utterance.append(speech_rate_each)

        output_pho_utterance, num_output_pho_utterance = remove_punc_esp_2_phonemes(utterance)
        num_output_pho.append(num_output_pho_utterance)
        output_pho.append(output_pho_utterance)
    # print("number of sequences:", len(duration_utterance)) # 122

    # Use .loc[row_indexer,col_indexer] = value to add new columns
    # When setting values in a pandas object, care must be taken to avoid what is called chained indexing. See:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    df_dialogue_no_none_new = df_dialogue_no_none.copy()
    df_dialogue_no_none_new.loc[:, 'duration_utterance'] = duration_utterance
    df_dialogue_no_none_new.loc[:, 'num_token_per_utterance'] = num_token_per_utterance
    df_dialogue_no_none_new.loc[:, 'speech_rate_per_utterance'] = speech_rate_per_utterance
    df_dialogue_no_none_new.loc[:, 'output_pho'] = output_pho
    df_dialogue_no_none_new.loc[:, 'num_output_pho'] = num_output_pho

    return df_dialogue_no_none_new

def calculateLexicalDiversity(tokens): # tokens: all tokens after passing the sentences
    """
    The concept of lexical diversity refers to a measure of unique vocabulary usage. 
    The type-to-token ratio (TTR): the number of unique words (types, V ) are compared against the total number of words (tokens, N).
    
    Brunét’s Index (BI), is another measure of lexical diversity that has a weaker dependence on text length, 
    with a smaller value indicating a greater degree of lexical diversity.
    
    An alternative is also provided by Honoré’s Statistic (HS),
    which emphasizes the use of words that are spoken only once (denoted by V1)
    """
    
    N = len(tokens)
    V = len(set(tokens)) # unique tokens
    # type-to-token ratio (TTR), lexical diversity, measure of unique vocabulary usage
    lexical_diversity_TTR = round(V / N, 2) # TTR = V/N
    lexical_diversity_BI = round(N**(V**(-0.165)), 2)
    fdist = FreqDist(tokens)
    V1_tokens = []
    for word, freq in fdist.items():
        if freq == 1:
            V1_tokens.append(word)
    V1 = len(V1_tokens) # V1: words that are spoken only once
    lexical_diversity_HS = round(100*(math.log(N/(1-V1/V))), 2)
    return lexical_diversity_TTR, lexical_diversity_BI, lexical_diversity_HS

def calculateLexicalDensity(dialogue):
    """
    content density (CD)
    CD = # of verbs + nouns + adjectives + adverbs / N

    propositional density (P-density)
    number of expressed propositions (verbs, adjectives, adverbs, prepositions, and conjunctions) divided by the total number of words
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("gilf/french-camembert-postag-model")
    model = AutoModelForTokenClassification.from_pretrained("gilf/french-camembert-postag-model")

    from transformers import pipeline

    nlp_token_class = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    pos_taggers = nlp_token_class(dialogue) # limit of input length ???
    
    #verbs: V, VIMP(verbe imperatif), VINF(verbe infinitif), VPP(participe passé), VPR(participe présent), VS(subjonctif)
    verbs = ['V', 'VIMP', 'VINF', 'VPP', 'VPR', 'VS']
    #nouns: NC(nom commun), NPP(nom propre)
    nouns = ['NC', 'NPP']
    #adjectives: ADJ, ADJWH
    adjectives = ['ADJ', 'ADJWH']
    #adverbs: ADV, ADVWH
    adverbs = ['ADV', 'ADVWH']
    #prepositions: P, P+D(préposition + déterminant)
    prepositions = ['P', 'P+D']
    #conjunctions: CC(conjonction de coordination), CS(conjonction de subordination)
    conjunctions = ['CC', 'CS']

    # content density (CD)
    taggers = []
    for i in range(len(pos_taggers)):
        tagger = pos_taggers[i]['entity_group']
        taggers.append(tagger)

    #CD = # of verbs + nouns + adjectives + adverbs / N
    CD_list = verbs + nouns + adjectives + adverbs
    CD_taggers = [x for x in taggers if x in CD_list]    #if tagger in verbs|nouns|adjectives|adverbs:
    c_density_score = round(len(CD_taggers)/len(taggers), 2)
    
    # propositional density (P-density)
    #P-density = # of verbs + adjectives + adverbs + prepositions + conjunctions / N
    p_density_list = verbs + adjectives + adverbs + prepositions + conjunctions
    p_density_taggers = [x for x in taggers if x in p_density_list]
    p_density_score = round(len(p_density_taggers)/len(taggers), 2)
    return c_density_score, p_density_score

def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]

def getInfoFromDialogue(filename):
    df_csv = readCSVFile(filename)
    # df_dialogue_no_none = df_csv.loc[df_csv[7] != "None"]  # delete lines which are "None"
    print(len(df_csv))
    # Remove sequences that contain only diacritics
    #pattern = r'^\<[^>]+\>$'  # Matches strings that start with "<", contain one or more characters that are not ">", and end with ">"
    #df_dialogue_no_none = df_csv[~df_csv["content"].str.contains(pattern)]
    df_dialogue_no_none = filter_rows_by_values(df_csv, "content", ["<nv> ", "<nv>", "<?>", "<di>", "<fi>", "<ri><nv>"])
    print(len(df_dialogue_no_none))
    # To filter out rows where the content is NaN (Not a Number), use the pd.isna() function from pandas
    df_dialogue_no_none = df_dialogue_no_none[~pd.isna(df_dialogue_no_none["content"])]
    print(len(df_dialogue_no_none))
    dialogue = get_dialogue_content(df_dialogue_no_none)
    questions_list = find_questions(df_dialogue_no_none)

    df_dialogue_no_none = get_Speech_Info(df_dialogue_no_none)
    duration_utterance = df_dialogue_no_none['duration_utterance']
    print(duration_utterance.describe())
    print("number of sequences:", len(duration_utterance))
    print(duration_utterance)
    num_token_per_utterance = df_dialogue_no_none['num_token_per_utterance']
    speech_rate_per_utterance = df_dialogue_no_none['speech_rate_per_utterance']
    output_pho = df_dialogue_no_none['output_pho']
    num_output_pho = df_dialogue_no_none['num_output_pho']

    # Passing the string text into word tokenize for breaking the sentences
    token = word_tokenize(dialogue, language="french")

    fdist = FreqDist(token) # FreqDist({'un': 3, 'très': 3, 'de': 3, 'dans': 3, "j'ai": 2, 'pas': 2, 'une': 2, 'du': 2, 'tout': 2, 'tu': 2, ...})
    num_tokens_unique = len(fdist) # Number of unique words spoken
    # To find the frequency of top 10 words
    # fdist_most_common = fdist.most_common(10)

    lexical_diversity_TTR, lexical_diversity_BI, lexical_diversity_HS = calculateLexicalDiversity(token)
    c_density_score, p_density_score = calculateLexicalDensity(dialogue[:512])

    # In the final version, only the following elements were used: num_tokens_unique, duration_utterance, num_token_per_utterance, num_output_pho, lexical_diversity_TTR
    return {"dialogue": dialogue,
            "questions_list": questions_list,
            "num_tokens_unique": num_tokens_unique,
            "duration_utterance": duration_utterance,
            "num_token_per_utterance": num_token_per_utterance,
            "speech_rate_per_utterance": speech_rate_per_utterance,
            "output_pho": output_pho,
            "num_output_pho": num_output_pho,
            "lexical_diversity_TTR": lexical_diversity_TTR,
            "lexical_diversity_BI": lexical_diversity_BI,
            "lexical_diversity_HS": lexical_diversity_HS,
            "c_density_score": c_density_score,
            "p_density_score": p_density_score
    }

def create_word_clouds(dialogue, id_seance_num, display=False):
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import matplotlib.pyplot as plt
    import nltk.corpus  # sample text for performing tokenization
    from nltk.corpus import stopwords  # importing stopwords from nltk library
    # nltk.download('stopwords')

    a = set(stopwords.words('french'))
    # initializing adding elements
    add_stopwords = ["j'ai", "non", "oui", "c'était", "est", "l'ai", "être", " être", "être ", "j'avais", "c'est",
                     "peu", "quand", "être", "fois", "peut", "tout", "donc", "sais", "dont", "alors", "ben", "jusqu",
                     "encore", "après", "qu'avec", "faut", "quel", "sous", "déjà", "qu'il", "quel", "près"]
    # update() appends element in set
    # internally reorders
    a.update(add_stopwords)
    dialogue1 = word_tokenize(dialogue.lower(), language="french")
    token_without_sw = [word for word in dialogue1 if word not in a]
    filtered_sentence = (" ").join(token_without_sw)

    # lower max_font_size, change the maximum number of word and lighten the background:
    # min_word_length: the minimum number of letters a word must have to be included in the word cloud
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white",
                         min_word_length = 3).generate(filtered_sentence)
    if display:
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    # Save the image in the img folder:
    # id_seance_num = file_name.split('__', 2)[1]  # A01A_seance1_1512
    os.makedirs("./img/", exist_ok=True)
    wordcloud.to_file("./img/{0}.png".format(id_seance_num))



