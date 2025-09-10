# import torch
import numpy as np
import os
import torch
from python_speech_features import logfbank
from sklearn.feature_extraction.text import TfidfVectorizer
from speechbrain.lobes.features import Fbank
# from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
# from LMs import HuggingFaceBERT
import pickle

class Feats_Module(object):
    """
    Feats module to extract features on audio files.
        
    """
    def __init__(self, device="cpu", feat_type="MFB", norm=False, adapt=True, buffer_max=10*16000, vectorizer_path=""):
        self.feat_type = feat_type
        self.device = device
        self.norm = norm
        self.adapt = adapt
        self.past_feats = []
        self.max_past_feats = 25
        self.mean_past = []
        self.std_past  = []
        self.coef_past = 0.05
        self.vectorizer_path = vectorizer_path
        self.init_comp_feats()

    def init_comp_feats(self):
        getSourceHuggingFace = {
            "wav2vec2-xlsr":                    "voidful/wav2vec2-xlsr-multilingual-56",
            "wav2vec2-large-xlsr-53":           "facebook/wav2vec2-large-xlsr-53",
            "wav2vec2-large-xlsr-53-french":    "facebook/wav2vec2-large-xlsr-53-french",
            "wav2vec2-FR-2.6K-base":            "LeBenchmark/wav2vec2-FR-2.6K-base",
            "wav2vec2-FR-3K-base":              "LeBenchmark/wav2vec2-FR-3K-base",
            "wav2vec2-FR-3K-large":             "LeBenchmark/wav2vec2-FR-3K-large",
            "wav2vec2-FR-7K-base":              "LeBenchmark/wav2vec2-FR-7K-base",
            "wav2vec2-FR-7K-large":             "LeBenchmark/wav2vec2-FR-7K-large",
            "bert-base-multilingual-cased":     "bert-base-multilingual-cased",
            "roberta-base":                     "roberta-base",
            "roberta-large":                    "roberta-large",
        }
        if self.feat_type == "MFB":
            self.compute_features = Fbank(n_mels=80, left_frames=0, right_frames=0, deltas=False)
        if "logfbank" in self.feat_type:
            def comp_feats(sig):
                feats = logfbank(sig,16000,nfilt=40)
                feat_mean = np.mean(feats, 0)
                feat_std  = np.std(feats, 0)
                feat_stat = np.concatenate((feat_mean, feat_std),0)
                feats = [feat_stat]
                return np.array(feats)
            self.compute_features = comp_feats
        if "wav2vec2" in self.feat_type:
            source = getSourceHuggingFace[self.feat_type]
            dirname = os.path.dirname(__file__)
            save_path = os.path.join(dirname, "Models", "wav2vec2_models", source)
            self.compute_features = HuggingFaceWav2Vec2(source=source, save_path=save_path, output_norm=False, freeze=True, freeze_feature_extractor=True).to(self.device)
        if "bert" in self.feat_type:
            source = getSourceHuggingFace[self.feat_type]
            dirname = os.path.dirname(__file__)
            save_path = os.path.join(dirname, "Models", "bert_models", source)
            self.compute_features = HuggingFaceBERT(source=source, save_path=save_path, device=self.device, pooled=False, output_norm=False, freeze=True)
        if "tfidf" in self.feat_type:
            if self.vectorizer_path == "": return
            train_data = np.loadtxt(self.vectorizer_path, dtype='str', delimiter='\t', skiprows=0)
            self.vectorizer = TfidfVectorizer()
            _ = self.vectorizer.fit_transform(train_data[:,1])
            def comp_feats(sig):
                return self.vectorizer.transform(sig).toarray()
            self.compute_features = comp_feats

    def extract_feats(self, sig):
        tensorize = True
        if "bert" in self.feat_type: tensorize = False
        if "logfbank" in self.feat_type: tensorize = False
        if "tfidf" in self.feat_type: tensorize = False
        if tensorize:
            sig = torch.tensor(sig).float().to(self.device).unsqueeze(0)
        feats = self.compute_features(sig)
        if ("logfbank" in self.feat_type) and self.norm:
            self.past_feats.append(feats)
            while len(self.past_feats) > self.max_past_feats:
                self.past_feats.pop(0)
            if len(self.past_feats) > 2:
                mean = np.mean(self.past_feats, 0)
                std  = np.std (self.past_feats, 0)
                feats = (feats - mean) / std
        elif self.norm:
            mean = torch.mean(feats, dim=1).detach().data
            std  = torch.std (feats, dim=1).detach().data
            if self.adapt:
                if len(self.mean_past) == 0: self.mean_past = mean.clone()
                if len(self.std_past ) == 0: self.std_past  = std.clone()
                self.coef_new = 1 - self.coef_past
                mean = self.coef_past*self.mean_past + self.coef_new*mean
                std  = self.coef_past*self.std_past  + self.coef_new*std
                self.mean_past = mean.clone()
                self.mean_past = std.clone()
            feats = (feats - mean) / std
        return feats

    

