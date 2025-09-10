import os
import glob
import numpy as np
import pandas as pd
import random
import speechbrain as sb
import torch
from speechbrain.lobes.features import Fbank
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class Feature_extracter(nn.Module):
    """A class for extracting audio, video, and textual features.
    Also, for saving and loading the features to avoid re-processing when possible.

    Attributes
    ----------
    feat_title : str
        The title of the feature to be extracted, e.g. "mfb" for Mel-filter bank on audio, or "roberta-large" for the large RoBERTa model on text.
    save_dir : str
        The path referring to the folder containing video files of theradia data.
    feat_func: function
        The function that will be applied on top of feature extraction. For example, to turn torch tensors to numpy arrays: `feat_func = lambda x: x.numpy()`.
    **feat_params: dict
        The keyword arguments related to the feature extractor that is to be used. e.g. for "mfb" features, one can pass the argument `n_mels=80`.

    Methods
    -------
    init_feat()
        Initialises the feature extractor.
    get_MFB()
        Extracts Mel-Filter Bank (MFB) features. Activated when feat_title="mfb".
    extract(x)
        Extracts features based on the given input x.

    """

    def __init__(
        self,
        save_dir,
        feat_title,
        feat_func=lambda a: a,
        save_models_dir="",
        **feat_params,
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_models_dir = save_models_dir
        if self.save_models_dir == "":
            self.save_models_dir = save_dir
        self.feat_title = feat_title
        self.feats_dir = os.path.join(self.save_dir, self.feat_title)
        # if not os.path.exists(self.feats_dir): os.makedirs(self.feats_dir)
        self.feat_params = feat_params
        self.feat_func = feat_func
        self.init_feat()

    def init_feat(self):
        """Initialises the feature extractor."""
        if "mfb" in self.feat_title:
            self.feat_extractor = self.get_MFB()
        elif self.feat_title == "fau":
            # print("Warning: the FAU feature extraction is not implemented.")
            pass
        elif "tfidf" in self.feat_title:
            self.feat_extractor = self.get_tfidf()
        elif "w2v2" in self.feat_title.lower():
            self.feat_extractor = self.get_w2v2()
        elif "bert" in self.feat_title.lower():
            self.feat_extractor = self.get_BERT()
        elif "csv" in self.feat_title:
            pass
        else:
            print("Warning: the feature_title is not implemented.")

    def get_tfidf(self):
        """Extractor function for extracting Term Frequency - Inverse Document Frequency (TF-IDF) features. Activated when feat_title="tfidf"."""
        random.seed(self.feat_params["seed"])
        np.random.seed(self.feat_params["seed"])
        self.tfidf_vectorizer = TfidfVectorizer()
        _ = self.tfidf_vectorizer.fit_transform(self.feat_params["corpus"])

        def extractor(x):
            return self.tfidf_vectorizer.transform(x).toarray()

        return extractor

    def get_MFB(self):
        """Extractor function for extracting Mel-Filter Bank (MFB) features. Activated when feat_title="mfb"."""
        MFB = Fbank(n_mels=self.feat_params["n_mels"])
        return MFB

    def get_w2v2(self):
        # from huggingface_wav2vec import HuggingFaceWav2Vec2
        from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2

        save_path = os.path.join(
            self.save_models_dir, "HuggingFace", self.feat_params["source"]
        )
        W2V2 = HuggingFaceWav2Vec2(
            source=self.feat_params["source"],
            save_path=save_path,
            output_norm=False,
            freeze=self.feat_params["freeze"],
            freeze_feature_extractor=self.feat_params["freeze"],
        )
        return W2V2.to(self.feat_params["device"])

    def get_BERT(self):
        from huggingface_LMs import HuggingFaceBERT

        save_path = os.path.join(
            self.save_models_dir, "HuggingFace", self.feat_params["source"]
        )
        bert = HuggingFaceBERT(
            source=self.feat_params["source"],
            save_path=save_path,
            output_norm=False,
            freeze=self.feat_params["freeze"],
        )
        return bert

    def extract(self, x):
        """Extract feature based on given input
        This is the most basic method for extracting features that is used across all other related methods in this class
        """
        feats = self.feat_extractor(x)
        feats = self.feat_func(feats)
        return feats

    def extract_audio_feats(self, ID_list=[], wav_paths=[], override=False):
        """Extracts audio featues from a list of IDs and wav_paths
        Should pass feat_func = lambda x: x.squeeze(0).numpy() in __init__
        """
        if not os.path.exists(self.feats_dir):
            os.makedirs(self.feats_dir)
        for i, (ID, wav_path) in enumerate(zip(ID_list, wav_paths)):
            # print_progress_bar(i + 1, len(ID_list), prefix = f'Extracting {self.feat_title} features:', suffix = 'completed', length=50)
            save_path = os.path.join(self.feats_dir, f"{ID}.csv")
            if os.path.exists(save_path) and (not override):
                continue
            waves = sb.dataio.dataio.read_audio(wav_path)
            feats = self.extract(waves.unsqueeze(0))
            df = pd.DataFrame(feats)
            df.to_csv(save_path, index=None)

    def get_audio_feat(
        self, ID="", wav_path="", waves=[], override=True, save_feats=True
    ):
        """Extracts audio featue based on an ID and a wav_path.
        The ID helps to store the feature and thus avoid re-calulating.
        """
        save_path = os.path.join(self.feats_dir, f"{ID}.csv")
        if "w2v2" in self.feat_title.lower():
            if not self.feat_params["freeze"]:
                override = True
        if os.path.exists(save_path) and (not override):
            feats = pd.read_csv(save_path, index_col=None).to_numpy()
            feats = torch.tensor(feats)
            if "w2v2" in self.feat_title.lower():
                feats = feats.to(self.feat_params["device"])
        else:
            if wav_path != "":
                waves = sb.dataio.dataio.read_audio(wav_path).unsqueeze(0)
            if "w2v2" in self.feat_title.lower():
                waves = waves.to(self.feat_params["device"])
            feats = self.extract(waves)
            if save_feats:
                if not os.path.exists(self.feats_dir):
                    os.makedirs(self.feats_dir)
                df = pd.DataFrame(feats.cpu().detach())
                df.to_csv(save_path, index=None)
        return feats

    def get_bert_feat(self, ID_list=[], trs_list=[], override=False, save_feats=False):
        """Extracts bert featue based on an ID and transcription list.
        The ID helps to store the feature and thus avoid re-calulating.
        """
        all_feats = []
        if not self.feat_params["freeze"]:
            override = True
        ID = ID_list[-1]
        paths_exist = True
        for ID in ID_list:
            save_path = os.path.join(self.feats_dir, f"{ID}.csv")
            if not os.path.exists(save_path):
                paths_exist = False
        if paths_exist and (not override):
            for ID in ID_list:
                save_path = os.path.join(self.feats_dir, f"{ID}.csv")
                feats = pd.read_csv(save_path, index_col=None).to_numpy()
                feats = torch.tensor(feats)
                feats = feats.to(self.feat_params["device"])
                all_feats.append(feats)
            feats = pad_sequence(all_feats, batch_first=True, padding_value=0.0)
        else:
            # print(trs_list)
            feats = self.extract(trs_list)
            # print(feats.size())
            if save_feats:
                if not os.path.exists(self.feats_dir):
                    os.makedirs(self.feats_dir)
                for i, ID in enumerate(ID_list):
                    save_path = os.path.join(self.feats_dir, f"{ID}.csv")
                    df = pd.DataFrame(feats.cpu().detach()[i])
                    df = df[(df.T != 0).any()]  # remove padded zeros
                    df.to_csv(save_path, index=None)
        return feats

    def get_fau_feat(self, ID="", video_path="", override=False, save_feats=False):
        """Get a fau feature"""
        save_path = os.path.join(
            self.feat_params["fau_dir"], f"{ID}.csv"
        )  # removed "self.feats_dir, "
        if os.path.exists(save_path) and (not override):
            feats = pd.read_csv(save_path, index_col=None).to_numpy()
        else:
            print(
                "FAU extraction is not implemented."
            )  # could not find a package for it, which does not require installation without sudo.
        return feats[:, 5:]

    def get_csv_feat(self, IDs=[], override=False, save_feats=False):
        """Get features from a csv file"""
        feats_all = []
        for ID in IDs:
            csv_dir = os.path.join(self.feat_params["csv_dir"], "**", f"{ID}.csv")
            csv_path = glob.glob(csv_dir, recursive=True)[0]
            feats = pd.read_csv(csv_path, index_col=None).to_numpy()
            feats = self.feat_func(feats)
            feats = torch.tensor(feats)  # .unsqueeze(1)
            feats_all.append(feats)
        feats_all = pad_sequence(feats_all, batch_first=True, padding_value=0.0)
        return feats_all

    def get_feat_size(self):
        """Get the size of the feature"""
        if "mfb" in self.feat_title:
            return self.feat_params["n_mels"]
        elif self.feat_title == "fau":
            return 35
        elif "tfidf" in self.feat_title:
            feats = self.extract([""])[0]
            return len(feats)
        elif "w2v2" in self.feat_title.lower():
            feats = self.extract(torch.rand(1, 1000).to(self.feat_params["device"]))
            # print(feats.shape)
            return feats.size()[1]
        elif "bert" in self.feat_title.lower():
            feats = self.extract([""])
            return feats.size()[-1]
        elif "csv" in self.feat_title:
            return self.feat_params["feats_size"]

    def forward(self, **params_dict):
        """Get a feature based on a params_dict, which contains necessary info for extracting a feature based on feature_title"""
        if "mfb" in self.feat_title or "w2v2" in self.feat_title.lower():
            feats = []
            for ID, wav_path in zip(params_dict["ID"], params_dict["wav_path"]):
                feat = self.get_audio_feat(ID, wav_path)
                # feat = torch.tensor(feat)
                feats.append(feat)
            # feats = torch.stack(feats, 0)
            feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
            # print(feats.size())
        elif self.feat_title == "fau":
            # feats = self.get_fau_feat(params_dict["ID"])
            feats = []
            for ID in params_dict["ID"]:
                feat = self.get_fau_feat(ID)
                feat = torch.tensor(feat)
                feats.append(feat)
            # feats = torch.concat(feats, 0)
            feats = pad_sequence(feats, batch_first=True, padding_value=0.0)
        elif "tfidf" in self.feat_title:
            feats = self.extract(params_dict["trs"])
            feats = torch.tensor(feats).unsqueeze(1)  # tfidf does not have sequence
            # feats = feats[0]
        elif "bert" in self.feat_title.lower():
            feats = self.get_bert_feat(params_dict["ID"], params_dict["trs"])
            # feats = self.extract(params_dict["trs"])
        elif "csv" in self.feat_title.lower():
            feats = self.get_csv_feat(params_dict["ID"])
        # print("feats", feats)
        return feats
