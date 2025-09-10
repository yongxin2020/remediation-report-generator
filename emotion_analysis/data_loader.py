import os
import glob
import json
import numpy as np
import pandas as pd
import random

def print_progress_bar(iteration: int, total: int, prefix = '', suffix = '', decimals = 1, length = "fit", fill = '█') -> None:
    """Prints a progress bar on the terminal
    useful to visualise the progress of loops 
    
    Example
    -------
    for i in range(100):
        print_progress_bar(i + 1, 100, prefix = f'Looping:', suffix = 'completed', length=50)
    
    """
    if length=="fit":
        rows, columns = os.popen('stty size', 'r').read().split() # checks how wide the terminal width is
        length = int(columns) // 2
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: # go to new line when the progress bar is finished
        print()

class Theradia_data_loader():
    """Theradia data loader for machine learning purposes.
    This class can load and preprocess Theradia data, and save/load them in json format.
    
    Attributes
    ----------
    videos_folder : str
        The path referring to the folder containing video files of theradia data.
    annotation_folder : str
        The path referring to the folder containing annotations corresponding to the video files.
    transcription_folder : str
        The path referring to the folder containing transcriptions corresponding to the video files.
    save_dir : str
        The path referring to the folder containing the processed data.
    json_path : str
        The path referring to the json file that is used for storing and loading processed data.
    data : dict
        Dictionary containing all the data related to each video, its path, emotion annotations and transcriptions.
    
    Methods
    -------
    index_data()
        Based on the paths, it will loads the paths of video files, and annotations as data. First step in preprocessing.
    get_csv_trans(video_file, trans_paths)
        Gets the transcription from the csv files that are related to the video files.
    get_csv_folder_given_name(video_name, annot_folds)
        Gets annotation csv file directory based on a video file path.
    extract_wavs()
        Extract wav files 16b integer mono from video files for audio processing.
    save_to_json()
        Saves the data in json format to the json_path.
    load_from_json()
        Loads the data from the json_path to python dict.
    add_labels_to_dict()
        Adds the emotion annotations to the data. This should be run after indexing the paths with the `index_data` method.
    black_list_IDs(ID_list)
        black-list a certain list of IDs.
    white_list_IDs(ID_list)
        only keep a certain list of IDs.
    auto_partition(split=[.75, .15, .15], seed=0)
        Automatically partition data based on a train-dev-test split. The seed is used to make partitioning reproducible.
    """
    
    def __init__(self, 
                 videos_folder, 
                 annotation_folder, 
                 transcription_folder,
                 save_dir,
                 json_name="data.json"):
        '''
        Parameters
        ----------
        videos_folder : str
            The path referring to the folder containing video files of theradia data.
        annotation_folder : str
            The path referring to the folder containing annotations corresponding to the video files.
        transcription_folder : str
            The path referring to the folder containing transcriptions corresponding to the video files.
        save_dir : str
            The path referring to the folder containing the processed data.
        json_name : str
            The name of the json file to save the processed data.
        '''
        self.videos_folder = videos_folder
        self.annotation_folder = annotation_folder
        self.transcription_folder = transcription_folder
        self.save_dir = save_dir
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        self.json_path = os.path.join(self.save_dir, json_name)
        self.partition_dir = os.path.join(self.save_dir, "partitions")
        self.wav_sample_rate = 16000
        self.data = {}
        self.labels_list = ['angry', 'annoyed', 'anxious', 'ashamed', 'confident',
                            'contemptuous', 'curious', 'desperate', 'disappointed',
                            'embarrassed', 'excited', 'frustrated', 'guilty', 'happy',
                            'hopeful', 'impatient', 'interested', 'nervous', 'proud',
                            'relaxed', 'sad', 'satisfied', 'surprised', 'tired', 'upset']
        self.dims_list = ["dimension_arousal", "dimension_novelty", "dimension_goal_conduciveness", 
                          "dimension_intrinsic_pleasantness", "dimension_coping"]
        self.dims_summ = ["arousal", "novelty", "goal conduciveness", "intrinsic pleasantness", "coping"]
        # self.index_data()
        
    def index_data(self):
        '''Based on the paths, it will loads the paths of video files, and annotations as data. First step in preprocessing.
        '''
        video_files = glob.glob(f"{self.videos_folder}/**/*.mp4", recursive=True)
        annot_folds = glob.glob(f"{self.annotation_folder}/**/*/", recursive=False)
        trans_paths = glob.glob(f"{self.transcription_folder}/**/*.csv", recursive=True)
        for video_file in video_files:
            video_name = os.path.basename(video_file)
            video_name = os.path.splitext(video_name)[0]
            seance_name = os.path.split(os.path.dirname(video_file))[1]
            ID = video_name
            annots_path = self.get_csv_folder_given_name(video_name, annot_folds)
            trans  = self.get_csv_trans(video_file, trans_paths)
            annots_path = os.path.join(self.annotation_folder, seance_name)
            datum = {
                "ID": ID,
                "video_path": video_file,
                "annots_path": {
                    "labels": os.path.join(annots_path, f"{ID}_labels.csv"),
                    "dimension_arousal": os.path.join(annots_path, f"{ID}_dimension_arousal.csv"),
                    "dimension_novelty": os.path.join(annots_path, f"{ID}_dimension_novelty.csv"),
                    "dimension_goal_conduciveness": os.path.join(annots_path, f"{ID}_dimension_goal_conduciveness.csv"),
                    "dimension_intrinsic_pleasantness": os.path.join(annots_path, f"{ID}_dimension_intrinsic_pleasantness.csv"),
                    "dimension_coping": os.path.join(annots_path, f"{ID}_dimension_coping.csv"),
                    "dimension_summaries": os.path.join(annots_path, f"{ID}_dimension_summaries.csv"),
                },
                "trs": trans,
            }
            self.data[ID] = datum
    
    def get_csv_trans(self, video_file, trans_paths):
        '''Gets the transcription from the csv files that are related to the video files.
        '''
        folder_id = os.path.split(os.path.split(video_file)[0])[1]
        for trans_path in trans_paths:
            trans_id = os.path.split(trans_path)[1]
            if folder_id in trans_id:
                df = pd.read_csv(trans_path, sep='\t', header=None)
                for row in df.iterrows():
                    file_name = row[1][0]
                    trans = row[1][1]
                    if os.path.basename(video_file) in file_name:
                        return trans[1:-1]
        return ""

    def get_csv_folder_given_name(self, video_name, annot_folds):
        '''Gets annotation csv file directory based on a video file path.
        '''
        for folder in annot_folds:
            if video_name in folder:
                return folder
        return ""
    
    def extract_wavs(self):
        '''Extract wav files 16b integer mono from video files for audio processing.
        '''
        data_keys = list(self.data.keys())
        for i, k in enumerate(data_keys):
            # print(self.data[k])
            print_progress_bar(i + 1, len(data_keys), prefix = f'Writting wav files:', suffix = 'completed', length=50)
            video_path = self.data[k]["video_path"]
            video_name = os.path.basename(video_path)
            seance_name = os.path.split(os.path.split(video_path)[0])[1]
            # print(seance_name, video_name)
            wav_dir = os.path.join(self.save_dir, "Audio", seance_name)
            if not os.path.exists(wav_dir): os.makedirs(wav_dir)
            wav_path = os.path.join(wav_dir, video_name.replace(".mp4", ".wav"))
            self.data[k]["wav_path"] = wav_path
            if os.path.exists(wav_path): continue # ignoring the wav files that already exist!
            arg = f'ffmpeg -i {video_path} -ar {self.wav_sample_rate} -ac 1 -c:a pcm_s16le -af "volume=0dB" -hide_banner -v 0 -y {wav_path}'
            os.system(arg)
    
    def save_to_json(self):
        '''Saves the data in json format to the json_path.
        '''
        with open(self.json_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.data, json_file)
            
    def load_from_json(self):
        '''Loads the data from the json_path to python dict.
        '''
        with open(self.json_path, 'r', encoding='utf-8') as json_file: 
            self.data = json.load(json_file)

    def add_labels_to_dict(self):
        '''Adds the emotion annotations to the data. This should be run after indexing the paths with the `index_data` method.
        '''
        data_labelled = self.data.copy()
        data_keys = list(self.data.keys())
        for i, k in enumerate(data_keys):
            print_progress_bar(i + 1, len(data_keys), prefix = f'Adding labels:', suffix = 'completed', length=50)
            annots_list = list(self.data[k]["annots_path"].keys())
            data_labelled[k]["annots"] = {}
            for annot in annots_list:
                annot_path = self.data[k]["annots_path"][annot]
                try:
                    df_annot = pd.read_csv(annot_path)
                    data_labelled[k]["annots"][annot] = [list(df_annot.columns)] + df_annot.to_numpy().tolist()
                except:
                    continue
        return data_labelled
    
    def black_list_IDs(self, ID_list=[]):
        '''black-list a certain list of IDs 
        Returns the filtered data (does not change self.data in-place)
        '''
        dict_filtered = {}
        for ID, dic in self.data.items():
            should_add = True
            for target_id in ID_list:
                if target_id in ID: should_add = False
            if should_add: 
                dict_filtered.update({ID:dic})
        return dict_filtered
    
    def white_list_IDs(self, ID_list=[]):
        '''only keep a certain list of IDs
        Returns the filtered data (does not change self.data in-place)
        '''
        dict_filtered = {}
        for ID, dic in self.data.items():
            should_add = False
            for target_id in ID_list:
                if target_id in ID: should_add = True
            if should_add: 
                dict_filtered.update({ID:dic})
        return dict_filtered
    
    def save_partitions(self, part_name="default"):
        '''Save partitioning IDs in a csv file. 
        Useful for having the same partitions across different experiments.
        This function will also generate a statistic file related to the partitions.
        '''
        if not os.path.exists(self.partition_dir): os.makedirs(self.partition_dir)
        part_path       = os.path.join(self.partition_dir, f"{part_name}.csv")
        part_stats_path = os.path.join(self.partition_dir, f"{part_name}_stats.csv")
        df = pd.DataFrame(self.partitions)
        df.to_csv(part_path, index=False)
        parts = list(np.unique(self.partitions["partition"]))#["train", "dev", "test"]
        stats = {"partition": parts, "count":[0 for _ in parts], "duration (s)":[0.0 for _ in parts]}
        for key, part, dur in zip(self.partitions["key"], self.partitions["partition"], self.partitions["duration (s)"]):
            idx = parts.index(part)
            stats["count"][idx] += 1
            stats["duration (s)"][idx] += dur
        stats["duration (s)"] = [round(v, 3) for v in stats["duration (s)"]]
        df_stats = pd.DataFrame(stats)
        df_stats.to_csv(part_stats_path, index=False)
        
    def load_partitions(self, part_name="default"):
        '''Load partitioning IDs from a CSV file, and return related train, dev, test dicts. 
        Useful for having the same partitions across different experiments.
        '''
        part_path = os.path.join(self.partition_dir, f"{part_name}.csv")
        df = pd.read_csv(part_path)
        keys  = df["key"].to_numpy()
        parts = df["partition"].to_numpy()
        all_keys = {part:[] for part in parts}
        for key, part in zip(keys, parts):
            all_keys[part].append(key)
        all_dicts = []
        for keys in list(all_keys.values()):
            part_dict = {key: self.data[key] for key in keys}
            all_dicts.append(part_dict)
        return all_dicts
    
    def merge_label_for_item(self, datum, merge_groups):
        '''Merge different labels annotations and add it as a new label group. 
        The datum refers to one item of data (i.e. the value of one key).
        The merge_groups should be a dict of str:list, containing new name for merge as str, and list of labels to be merged.
        '''
        all_labels = np.array(datum["annots"]["labels"])[:, 1].tolist()
        for merge_name, merge_group in merge_groups.items():
            if merge_name in all_labels: continue
            annots_to_merge = []
            for label in merge_group:
                idx = all_labels.index(label)
                annots = np.array(datum["annots"]["labels"])[idx, 2:].astype(float)
                annots_to_merge.append(annots)
            annots_to_merge = np.array(annots_to_merge)
            merge_annots = list(np.mean(annots_to_merge, 0))
            datum["annots"]["labels"].append([merge_name, merge_name]+merge_annots)
    
    def merge_labels(self, merge_groups):
        '''Merge different labels annotations for all data and add it as a new label group. 
        The merge_groups should be a dict of str:list, containing new name for merge as str, and list of labels to be merged.
        '''
        for i, (k, datum) in enumerate(self.data.items()):
            print_progress_bar(i + 1, len(self.data), prefix = f'Merging labels:', suffix = 'completed', length=50)
            self.merge_label_for_item(datum, merge_groups)
            
    def filter_data_with_criteria(self, criteria): # change this to only keep annots of certain user_IDs
        '''Filter data with a given criteria (function which returns True, or False given a datum)
        '''
        new_data = {}
        for i, (k, datum) in enumerate(self.data.items()):
            print_progress_bar(i + 1, len(self.data), prefix = f'Filtering data:', suffix = 'completed', length=50)
            # try:
            if criteria(datum): new_data[k] = datum
            # except:
            #     pass
        self.data = new_data
        
    def filter_data_with_users(self, users):
        '''Only keeping annotations of the annotators that are within the list of users
        '''
        def criteria(datum):
            dims = ["dimension_summaries", "dimension_arousal", "dimension_novelty", "dimension_goal_conduciveness", "dimension_intrinsic_pleasantness", "dimension_coping"]
            valid = True
            all_users = []
            all_users.append(datum["annots"]["labels"][0][2:])
            for user1 in datum["annots"]["labels"][0][2:]:
                if not user1 in users:
                    idx = datum["annots"]["labels"][0].index(user1)
                    for i in range(len(datum["annots"]["labels"])):
                        del datum["annots"]["labels"][i][idx]
            for dim in dims:
                try:
                    users_dim = datum["annots"][dim][0][1:]
                    all_users.append(users_dim)
                except:
                    valid = False; break
                for user1 in users:
                    for user_list in all_users:
                        if not user1 in user_list: valid = False; break
                        for u, user_l in enumerate(user_list): # removing extra user annotations
                            if (not user_l in users) and (user_l in datum["annots"][dim][0]):
                                idx = datum["annots"][dim][0].index(user_l)
                                for i in range(len(datum["annots"][dim])):
                                    del datum["annots"][dim][i][idx]
                            if "user_1" in datum["annots"][dim][0]:
                                idx = datum["annots"][dim][0].index("user_1")
                                for i in range(len(datum["annots"][dim])):
                                    del datum["annots"][dim][i][idx]
                                # del datum["annots"][dim][0][idx]
            #                 print("ID", datum["ID"], user_l, valid)
            # if datum["ID"] == "A01A_1_049":
            #     print("valid", valid)
            return valid
        self.filter_data_with_criteria(criteria)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]