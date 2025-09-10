import os
import torch
import numpy as np
#import pyaudio
from threading import Thread
import pandas as pd
from modules_feats import module_tfidf, module_mfb, module_bert, module_w2v2
from VAD_Feature import Feats_Module
from VAD_Module import VAD_Module
from tinydb import TinyDB, Query
import speech_recognition
import time
from scipy.io import wavfile


class Backend_audio:
    def __init__(self, settings=None):
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.file_dir, "data")
        self.models_dir = os.path.join(self.file_dir, "models")
        self.affect_labels = [
            "annoyed",
            "anxious",
            "confident",
            "desperate",
            "frustrated",
            "happy",
            "interested",
            "relaxed",
            "satisfied",
            "surprised",
        ]
        # self.settings = TinyDB(os.path.join(self.file_dir, "data", "settings.json"), ensure_ascii=False)
        if settings is None:
            self.get_settings()
        else:
            self.settings = settings
            try:
                self.settings_vad = self.settings["VAD"]
            except:
                pass
        logs_dir = os.path.join(self.file_dir, "logs")
        session_id = self.settings["session_id"]
        self.device = self.settings["device"]
        self.label_models = {}
        self.dim_model_sum = {}
        self.dim_model_con = {}
        self.audio_feat_title = self.settings["features"]["audio"]
        self.text_feat_title = self.settings["features"]["text"]
        self.logger = self.settings["logger"]  # False # read from settings
        if self.logger:
            self.db_dir = os.path.join(logs_dir, str(session_id))
            if not os.path.exists(self.db_dir):
                os.makedirs(self.db_dir)
            self.db = TinyDB(
                os.path.join(self.db_dir, "backend_audio.json"), ensure_ascii=False
            )
        self.sr = speech_recognition.Recognizer()
        self.init_models()
        self.segment_save_path = ""
        self.turn_counter = 0

        def callback():
            pass

        self.call_back_fn = callback
        try:
            self.turn_counter = self.db.search(Query().turn >= 0)[-1]["turn"]
            self.turn_counter += 1
        except:
            pass

    def get_settings(self):
        self.settings_db = TinyDB(
            os.path.join(self.file_dir, "data", "settings.json"), ensure_ascii=False
        )
        self.settings = self.settings_db.all()[0]
        self.settings_vad = self.settings["VAD"]

    def init_vad(self):
        self.get_settings()
        self.vad_model_path = os.path.join(
            self.file_dir, "models", "VAD", "gru_model_jit.pth"
        )
        self.VAD_feats = Feats_Module(device=self.device, feat_type="MFB", norm=True)
        self.VAD_model = VAD_Module(device=self.device, model_path=self.vad_model_path)
        self.prep_to_listen()
        if self.settings_vad["listen"]:
            T = Thread(target=self.listen_loop, args=())
            T.start()
            # self.listen_loop()

    def prep_to_listen(self):
        self.settings_vad["listen"] = True
        self.settings_db.update({"VAD": self.settings_vad})
        self.frame_rate = self.settings_vad["frame_rate"]
        self.buffer_max = (
            self.settings_vad["buffer_sec"] * self.settings_vad["frame_rate"]
        )
        p = pyaudio.PyAudio()
        self.CHUNK = int(self.settings_vad["chunk_sec"] * self.frame_rate)  # 2**15
        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.frame_rate,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        self.num_of_chunks = (
            self.settings_vad["listen_time"] / self.settings_vad["chunk_sec"]
        )
        print("VAD is prepared to listen")

    def listen_loop(self):
        self.buffer = np.array([])
        for i in range(int(self.num_of_chunks)):
            print("Listening", i, time.time())
            self.get_settings()
            # audioChunk = np.fromstring(self.stream.read(self.CHUNK, exception_on_overflow = False),dtype=np.int16)
            audioChunk = np.frombuffer(
                self.stream.read(self.CHUNK, exception_on_overflow=False),
                dtype=np.int16,
            )
            # np.frombuffer(astr.encode(),'u1')
            sig = audioChunk / 32767.0
            self.buffer = np.concatenate((self.buffer, sig), 0)
            if len(self.buffer) > self.buffer_max:
                self.buffer = self.buffer[-self.buffer_max :]

            segment_detected = False
            self.segment = self.buffer
            self.feats_VAD = self.VAD_feats.extract_feats(self.buffer)
            feats_fs = (len(self.buffer) / self.frame_rate) / self.feats_VAD.size()[1]
            preds = self.VAD_model.predict(self.feats_VAD)
            times = self.VAD_model.post_process(
                feats_fs=feats_fs,
                hys_top=self.settings_vad["hys_top"],
                hys_bottom=self.settings_vad["hys_bottom"],
                cutWin_sec=self.settings_vad["cutWin_sec"],
                mergeWin_sec=self.settings_vad["mergeWin_sec"],
            )
            if len(times) > 0:
                start = int(len(self.buffer) * times[0][0]) - int(0.5 * self.frame_rate)
                stop = int(len(self.buffer) * times[-1][-1])
                coeff = self.settings_vad["chunk_sec"]
                checkTime = len(self.buffer) - coeff * len(sig)
                if not stop > checkTime:  # wait a chunk to decide!
                    self.segment = self.buffer[start:-1]
                    segment_detected = True

            if segment_detected:
                # dic = {"segment":self.segment}
                T = Thread(target=self.process_detected_segment, args=(self.segment,))
                T.start()
                T = Thread(target=self.listen_loop, args=())
                T.start()
                break

            if not self.settings_vad["listen"]:
                break

    def process_detected_segment(self, segment):
        print("segment detected")
        time_end = time.time()
        trs = self.get_trs()
        if trs == "":
            return
        if self.settings["save_wav"]:
            self.save_wav(segment)
        audio_preds, audio_preds_dims = self.predict_models_audio(
            segment, dir=self.audio_feat_title
        )
        text_preds, text_preds_dims = self.predict_models_tfidf(
            trs, dir=self.text_feat_title
        )
        audio_cls = self.classify_preds([audio_preds], [self.audio_feat_title])
        text_cls = self.classify_preds([text_preds], [self.text_feat_title])
        text_audio_cls = self.classify_preds(
            [text_preds, audio_preds], [self.text_feat_title, self.audio_feat_title]
        )
        # print("trs:", trs)
        # print("audio_preds", audio_preds)
        # print("text_preds", text_preds)
        # now you have to save them in a file with timestamps!
        if self.logger:
            duration = float(len(segment) / 16000)
            time_start = time_end - duration
            self.save_preds(
                trs=trs,
                time_start=time_start,
                time_end=time_end,
                duration=duration,
                text_feat_title=self.text_feat_title,
                audio_feat_title=self.audio_feat_title,
                text_preds=text_preds,
                audio_preds=audio_preds,
                text_preds_dims=text_preds_dims,
                audio_preds_dims=audio_preds_dims,
                text_cls=text_cls,
                audio_cls=audio_cls,
                text_audio_cls=text_audio_cls,
            )
        self.call_back_fn()

    def classify_preds(self, preds_dicts, feat_tiltes):
        preds_all = []
        print("preds_dicts", preds_dicts)
        for preds_dict in preds_dicts:
            preds_all.append(np.array(list(preds_dict.values())))
        preds = np.concatenate(preds_all, 1)
        pre = "class_LinReg_1_"
        csv_name = pre + "_".join(feat_tiltes) + ".csv"
        csv_path = os.path.join(self.data_dir, "decision_fusion", csv_name)
        df = pd.read_csv(csv_path)
        coefs = df.to_numpy()[:, 5:]
        bias = df.to_numpy()[:, 4]
        thresholds = df.to_numpy()[:, 3]
        # print(csv_path)
        # print("coefs", coefs.shape)
        # print("preds", preds.shape)
        # print("bias", bias)
        probs = np.sum(preds * coefs, 1) + 100 * bias
        # influences = []
        # for i in range(len(feat_tiltes)):
        #     inf = np.sum(preds[:, (i*6):(i*6+6)] * coefs, 1)
        #     influences.append(int(inf))
        thresholds = 100 * thresholds
        classes = {}
        for prob, thr, label in zip(probs, thresholds, self.affect_labels):
            # print(label, prob, thr)
            cls = 0 if prob < thr else 1
            classes[label] = cls
            influences = []
            for i in range(len(feat_tiltes)):
                l = self.affect_labels.index(label)
                inf = np.sum(
                    preds[l, (i * 6) : (i * 6 + 6)] * coefs[l, (i * 6) : (i * 6 + 6)]
                )
                influences.append(inf)
            influences = np.array(influences)
            influences = np.abs(influences) / np.sum(np.abs(influences))
            classes[f"{label}_influence"] = list(influences)
        # print(classes)
        return classes
        # print("probs", probs)

    def save_wav(self, segment):
        save_dir = os.path.join(self.db_dir, "wavs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.segment_save_path = os.path.join(
            self.db_dir, "wavs", f"turn_{self.turn_counter}.wav"
        )
        segment_int16 = (
            np.iinfo(np.int16).max * (segment / np.abs(segment).max())
        ).astype(np.int16)
        wavfile.write(self.segment_save_path, 16000, segment_int16)

    def save_preds(self, **kwargs):
        time_end = time.time()
        data_dict = {
            "turn": self.turn_counter,
            "wav_path": self.segment_save_path,
        }
        for kw, arg in kwargs.items():
            data_dict[kw] = arg
        self.db.insert(data_dict)
        print("audio_dict:", data_dict)
        self.turn_counter = self.turn_counter + 1

    def get_trs(self):
        trs = ""
        wav_bytes = (self.segment * 32767).astype(np.int16).tobytes()
        audio_data = speech_recognition.AudioData(
            wav_bytes, sample_rate=16000, sample_width=2
        )
        try:
            trs = self.sr.recognize_google(audio_data, language="fr-FR")
        except:
            trs = ""
        del audio_data
        return trs

    def init_models(self):
        # load feature models
        if "mfb" in self.audio_feat_title:
            self.audio_feat = module_mfb(
                feat_title=self.audio_feat_title, logger=self.logger
            )
        if "w2v2" in self.audio_feat_title:
            self.audio_feat = module_w2v2(
                feat_title=self.audio_feat_title, logger=self.logger
            )
        if "tfidf" in self.text_feat_title:
            self.text_feat = module_tfidf(
                feat_title=self.text_feat_title, logger=self.logger
            )
        if "bert" in self.text_feat_title:
            self.text_feat = module_bert(
                feat_title=self.text_feat_title, logger=self.logger
            )
        # load main models
        for dir in [self.audio_feat_title, self.text_feat_title]:
            models_dir = os.path.join(self.models_dir, dir)
            self.label_models[dir] = []
            for label in self.affect_labels:
                path = os.path.join(models_dir, f"{label}.pth")
                model = torch.jit.load(path, map_location=self.device)
                self.label_models[dir].append(model)
            path = os.path.join(models_dir, f"dim_sum.pt")
            self.dim_model_sum[dir] = torch.jit.load(path, map_location=self.device)

    def predict_models_tfidf(self, trs, dir="tfidf_google"):
        feats = self.text_feat.extract(trs)
        tensor = torch.tensor(feats).float().unsqueeze(0)
        if "tfidf" in self.text_feat_title:
            tensor = tensor.unsqueeze(0)
        preds = {label: 0 for label in self.affect_labels}
        # print(torch.tensor(feats).float().unsqueeze(0))
        # print(tensor.size())
        for label, label_model in zip(self.affect_labels, self.label_models[dir]):
            # print(label_model)
            pred = label_model(tensor)
            pred = pred.squeeze(0).detach().numpy()  # .mean().item()
            # preds[label] = [str(round(p, 3)) for p in pred]
            preds[label] = [int(p * 100) for p in pred]
        # print(tensor.size())
        pred = self.dim_model_sum[dir](tensor)
        pred = pred.squeeze().detach().numpy()
        preds_dims_sum = [round(float(p), 3) for p in pred]
        return preds, preds_dims_sum

    def predict_models_audio(self, audio_array, dir="mfb_normed"):
        segment_torch = torch.tensor(audio_array).float().unsqueeze(0)
        feats = self.audio_feat.extract(self.segment_save_path, segment_torch)
        preds = {label: 0 for label in self.affect_labels}
        # print(torch.tensor(feats).float().unsqueeze(0))
        for label, label_model in zip(self.affect_labels, self.label_models[dir]):
            tensor = torch.tensor(feats).float().unsqueeze(0)
            pred = label_model(tensor)
            # pred = pred.squeeze(0).mean().item()
            # preds[label] = round(pred, 3)
            pred = pred.squeeze(0).detach().numpy()  # .mean().item()
            preds[label] = [int(p * 100) for p in pred]
        pred = self.dim_model_sum[dir](tensor)
        pred = pred.squeeze().detach().numpy()
        preds_dims_sum = [round(float(p), 3) for p in pred]
        return preds, preds_dims_sum

    def stop_listening(self):
        self.settings_vad["listen"] = False
        self.settings_db.update({"VAD": self.settings_vad})


if __name__ == "__main__":
    from settings import Settings

    settings = Settings(session_id=0)
    settings.setup()
    backend = Backend_audio()
    txt = "super bien, très content"
    preds = backend.predict_models_tfidf(txt)
    print(f"predictions for {txt}:", preds)
    print(f"Testing VAD")
    backend.init_vad()
    # backend.stop_listening()
    print(f"VAD is working fine")
