from data_loader import Theradia_data_loader
import os
import json
import torch
import numpy as np
from feature_extractor import Feature_extracter
from tinydb import TinyDB, Query
import time


class modules:
    def __init__(self, logger=True):
        self.logger = logger
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.file_dir, "models")
        self.data_dir = os.path.join(self.file_dir, "data")
        logs_dir = os.path.join(self.file_dir, "logs")
        self.settings_db = TinyDB(
            os.path.join(self.data_dir, "settings.json"), ensure_ascii=False
        )
        session_id = self.settings_db.search(Query().session_id >= 0)[-1]["session_id"]
        self.save_dir = os.path.join(logs_dir, str(session_id))
        if self.logger:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)


class module_tfidf(modules):
    def __init__(self, feat_title="tfidf", **kwargs):
        super().__init__(**kwargs)
        if self.logger:
            self.db = TinyDB(
                os.path.join(self.save_dir, "tfidf.json"), ensure_ascii=False
            )
        self.dataset = Theradia_data_loader(
            "",
            "",
            "",
            self.data_dir,
            json_name=os.path.join(self.data_dir, "data_asr.json"),
        )

        self.dataset.load_from_json()
        train_dict, dev_dict, test_dict = self.dataset.load_partitions("train_dev_test")
        datum_key = "trs"
        if "google" in feat_title:
            datum_key = "trs_google"
        corpus = [datum[datum_key] for datum in list(train_dict.values())]
        self.feat_extractor = Feature_extracter(
            self.save_dir, "tfidf", corpus=corpus, seed=0
        )

    def extract(self, trs="", wav_path=""):
        time_start = time.time()
        feats = self.feat_extractor.extract([trs])[0]
        time_end = time.time()
        if self.logger:
            self.db.insert(
                {
                    "time_start": time_start,
                    "time_end": time_end,
                    "trs": trs,
                    "wav_path": wav_path,
                    "feats": list(feats),
                }
            )
        # print(self.db.search(Query().time > 1689936404)[-1]["time"])
        return feats


class module_bert(modules):
    def __init__(self, feat_title="bert_sentiment", device="cpu", **kwargs):
        super().__init__(**kwargs)
        if self.logger:
            self.db = TinyDB(
                os.path.join(self.save_dir, "bert.json"), ensure_ascii=False
            )

        self.feat_extractor = Feature_extracter(
            self.save_dir,
            feat_title,
            save_models_dir=self.models_dir,
            source="nlptown/bert-base-multilingual-uncased-sentiment",
            device=device,
            freeze=True,
            seed=0,
        )

    def extract(self, trs="", wav_path=""):
        time_start = time.time()
        feats_id = os.path.basename(wav_path).replace(".wav", "")
        if feats_id == "":
            feats_id = "temp"
        feats = self.feat_extractor.get_bert_feat(
            ID_list=[feats_id], trs_list=[trs], save_feats=self.logger, override=True
        )[0]
        feats = feats.numpy()
        # print(feats.numpy())
        time_end = time.time()
        if self.logger:
            self.db.insert(
                {
                    "time_start": time_start,
                    "time_end": time_end,
                    "trs": trs,
                    "wav_name": feats_id,
                    "wav_path": wav_path,
                    "feats_path": os.path.join(
                        self.save_dir, self.feat_extractor.feat_title, feats_id + ".csv"
                    ),
                }
            )
        return feats


class module_mfb(modules):
    def __init__(self, feat_title="mfb_normed", **kwargs):
        super().__init__(**kwargs)
        if self.logger:
            self.db = TinyDB(
                os.path.join(self.save_dir, "mfb.json"), ensure_ascii=False
            )
        # self.wavs_dir = os.path.join(self.save_dir, "wavs")
        n_mels = 80
        self.mean = np.array([0.0 for i in range(n_mels)])
        self.std = np.array([1.0 for i in range(n_mels)])
        self.feat_extractor = Feature_extracter(
            self.save_dir,
            feat_title,
            n_mels=80,
            feat_func=lambda x: (x.squeeze(0) - self.mean) / self.std,
        )
        self.feats_list = []
        self.feats_list_max = 20

    def extract(self, wav_path="", waves=[]):
        time_start = time.time()
        feats_id = os.path.basename(wav_path).replace(".wav", "")
        if feats_id == "":
            feats_id = "temp"
        feats = self.feat_extractor.get_audio_feat(
            ID=feats_id, wav_path=wav_path, waves=waves, save_feats=False
        )
        self.feats_list.append(feats)
        self.update_mean_std()
        # get feats again to have them normalised, but this time save them
        feats = self.feat_extractor.get_audio_feat(
            ID=feats_id, wav_path=wav_path, waves=waves, save_feats=self.logger
        )
        feats = feats.numpy()
        if self.logger:
            time_end = time.time()
            self.db.insert(
                {
                    "time_start": time_start,
                    "time_end": time_end,
                    "wav_name": feats_id,
                    "wav_path": wav_path,
                    "feats_path": os.path.join(
                        self.save_dir, self.feat_extractor.feat_title, feats_id + ".csv"
                    ),
                }
            )
        return feats

    def update_mean_std(self):
        while len(self.feats_list) > self.feats_list_max:
            del self.feats_list[0]
        feats_cated = np.concatenate(self.feats_list, 0)
        self.mean = feats_cated.mean(0)
        self.std = feats_cated.std(0)


class module_w2v2(modules):
    def __init__(self, feat_title="w2v2_xlsr", device="cpu", **kwargs):
        super().__init__(**kwargs)
        if self.logger:
            self.db = TinyDB(
                os.path.join(self.save_dir, "w2v2_xlsr.json"), ensure_ascii=False
            )
        # self.wavs_dir = os.path.join(self.save_dir, "wavs")
        self.feat_extractor = Feature_extracter(
            self.save_dir,
            feat_title,
            save_models_dir=self.models_dir,
            source="voidful/wav2vec2-xlsr-multilingual-56",
            device=device,
            freeze=True,
            feat_func=lambda x: x.squeeze(0),
        )

    def extract(self, wav_path="", waves=[]):
        time_start = time.time()
        feats_id = os.path.basename(wav_path).replace(".wav", "")
        if feats_id == "":
            feats_id = "temp"
        feats = self.feat_extractor.get_audio_feat(
            ID=feats_id, wav_path=wav_path, waves=waves, save_feats=self.logger
        )
        feats = feats.numpy()
        if self.logger:
            time_end = time.time()
            self.db.insert(
                {
                    "time_start": time_start,
                    "time_end": time_end,
                    "wav_name": feats_id,
                    "wav_path": wav_path,
                    "feats_path": os.path.join(
                        self.save_dir, self.feat_extractor.feat_title, feats_id + ".csv"
                    ),
                }
            )
        return feats


if __name__ == "__main__":
    print("testing tfidf module")
    tfidf = module_tfidf(logger=True)
    feats = tfidf.extract("salut, ça va super bien")
    print("tfidf feats length:", len(feats))
    # print("testing bert module")
    # bert = module_bert(logger=True, device="cpu")
    # feats = bert.extract("salut, ça va super bien")
    # print("bert feats length:", len(feats))
    print("testing mfb module")
    mfb = module_mfb(logger=True)
    feats = mfb.extract(os.path.join(mfb.data_dir, "test.wav"))
    feats = mfb.extract(os.path.join(mfb.data_dir, "test2.wav"))
    feats = mfb.extract(
        "", torch.rand(1, 16000 * 4)
    )  # testing from wav signal not a path
    print("testing mfb passed")
    # print("testing w2v2 module")
    # w2v2 = module_w2v2(logger=True, device="cpu")
    # feats = w2v2.extract(os.path.join(w2v2.data_dir, "test.wav"))
    # feats = w2v2.extract(os.path.join(w2v2.data_dir, "test2.wav"))
    # feats = w2v2.extract("", torch.rand(1, 16000*4)) # testing from wav signal not a path
    # print("testing w2v2 passed")
