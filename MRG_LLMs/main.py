"""
LLM-Based Report Generation Script

This script generates clinical reports using Large Language Models (GPT-3.5/GPT-4) 
from structured session variables.

Data Organization:
- INPUT: archive_json_variables/{participant}_seance{session}_{date}.json
  - Contains structured variables from the main generation pipeline
  - These are the processed session data (emotions, exercises, linguistics, etc.)
  
- OUTPUT: 
  - JSON responses: results/current/variables_4_{participant}_{session}.json
  - MD reports: results/current/{model}_{participant}_seance{session}.md  
  - HTML reports: results/current/{model}_{participant}_seance{session}.html

Usage:
    python main.py --code M01E --seance 1 --model gpt-4-0613
"""

from openai import OpenAI
from openai import AsyncOpenAI
import json
import argparse
import tqdm
import time
from preprocess import*
import sys
sys.path.append("../generate/")

import pandas as pd
import glob
import os
import yaml

# import markdownify

path_to_yaml = "../settings.yaml"

with open(path_to_yaml, 'r') as file:
    config = yaml.safe_load(file)

mainPath = config["mainPath"]

main_path = Path(config["paths"]["main"])
transcripts_dir = config["paths"]["transcripts"]  # "Transcripts and Logs"

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
    
# /data/depot/processed-theradia-data/M-subjects/M01E/M01E_seance1_0405/Transcripts and Logs
logsPath = os.path.join(mainPath, "{}/{}/{}/{}")
# transcriptionPath: / data / depot / processed - theradia - data / M - subjects / M01E / M01E_seance1_0405 / Transcripts and Logs
transcriptionPath = os.path.join(mainPath, "{}/{}/{}/{}")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    import tiktoken
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--code', type=str, default='M01E') # python ./main_phrases_trous.py "code:M01E;seance:1"
    argparser.add_argument('--seance', type=str, default='1')
    argparser.add_argument('--prompt_fp', type=str, default='prompts/JSON_variables.txt')
    argparser.add_argument('--save_fp', type=str, default='results/current/variables_4_M01E_1.json')
    argparser.add_argument('--key', type=str, required=True)
    argparser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613') #gpt-4-0613
    args = argparser.parse_args()
    
    OPENAI_API_KEY = args.key
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = open(args.prompt_fp).read()
    # Get the filename (prompt_type) without the extension from a path
    prompt_type = os.path.splitext(os.path.basename(args.prompt_fp))[0]

    # should modify this because the paths to csv and log have been changed (not now, cause not using guideline + dialogue/log)
    code = args.code
    seance = args.seance
    group = code[0]
    csv_name = f"{group}-subjects/{code}/{code}_seance{seance}_*/Transcripts and Logs/{code}_seance{seance}_*_audio_farfield_trs.csv"
    log_name = f"{group}-subjects/{code}/{code}_seance{seance}_*/Transcripts and Logs/{code}_seance{seance}_*.log"

    dialogue_files = glob.glob(os.path.join(mainPath, csv_name))
    log_file = glob.glob(os.path.join(mainPath, log_name))

    ct, ignore = 0, 0

    new_json = {}
    if prompt_type == 'guideline':
        # source_transcript = pd.read_csv(dialogue_file, delimiter="\t", header=None) #dataframe
        source_transcript = csv_dialog2txt(dialogue_file)
        # print("source_transcript", source_transcript) # <class 'str'>, only dialogue
        # log_file = instance['LogSample']['path']
        source_log = readLogs(log_file) # # <class 'str'>
        cleaned_source_log = logDataClean(source_log) # to reduce the log file length
        # print("source_log", source_log)
        cur_prompt = prompt.replace('{{Transcript}}', source_transcript).replace('{{Log}}', cleaned_source_log) # source_transcript and source_log: <class 'str'>
    elif prompt_type == 'JSON_variables':
        # Look for JSON variable files in archive_json_variables directory
        # These contain structured session data to be converted into reports
        json_pattern = f'./archive_json_variables/{code}_seance{seance}_*.json'
        json_files = glob.glob(json_pattern)

        # Check if any matching files were found
        if json_files:
            print(f"Found input variables file: {json_files[0]}")
            with open(json_files[0]) as json_data:
                source_variables = json.loads(json_data.read())
                # source_variables is initially a dict, need dict into string conversion using str() function
                source_variables=str(source_variables)
                cur_prompt = prompt.replace('{{Variables}}', source_variables)
                input_tokens_number = num_tokens_from_string(cur_prompt, args.model)
                print(f"Total number of input tokens are {input_tokens_number}")
        else:
            print(f"No matching JSON files found with pattern: {json_pattern}")
            print("Available files in archive_json_variables/:")
            available_files = glob.glob('./archive_json_variables/*.json')
            for f in available_files:
                print(f"  - {f}")
            raise FileNotFoundError(f"No input variables file found for {code} session {seance}")

    new_json["cur_prompt"]=cur_prompt

    try:
        # _response = openai.ChatCompletion.create(
        # new
        # client = AsyncOpenAI()
        _response = client.chat.completions.create( #await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": cur_prompt}],
            temperature=0,
            #max_tokens=5,
            #top_p=1,
            #frequency_penalty=0,
            #presence_penalty=0,
            #stop=None,
            # logprobs=40,
            #n=20
        )
        time.sleep(0.5)

        response = _response.choices[0].message.content #_response['choices'][0]['message']['content'] 
        # all_responses = [_response['choices'][i]['message']['content'] for i in
        #                  range(len(_response['choices']))]
        # instance['all_responses'] = all_responses
        new_json['response'] = response
        # new_json.append(instance)
        ct += 1
        # break
    except Exception as e:
        print(e)
        if ("limit" in str(e)):
            time.sleep(2)
        else:
            ignore += 1
            print('ignored', ignore)

            # break

    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
         json.dump(new_json, f, indent=4, ensure_ascii=False)
    # the following can be commented out
    with open(f'./results/current/{args.model}_{code}_seance{seance}.md', 'w', encoding='utf-8') as fp:
        fp.write(new_json['response'])

    os.system(
        'pandoc -s results/current/{}_{}_seance{}.md --metadata pagetitle="{}" -o results/current/{}_{}_seance{}.html'.format(args.model,
            code, seance, "Exemple participant_{}".format(code), args.model, code, seance))


