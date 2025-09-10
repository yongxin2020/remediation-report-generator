import os
import numpy as np
import pandas as pd
from tinydb import TinyDB, Query
from datetime import datetime
from backend_video import BackendVideo
import random


class Backend_main:
    def __init__(self):
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
        self.affect_pos = ["confident", "happy", "interested", "relaxed", "satisfied"]
        self.affect_neg = ["annoyed", "anxious", "desperate", "frustrated", "surprised"]
        self.dimensions = [
            "arousal",
            "coping",
            "goal_conduciveness",
            "intrinsic_pleasantness",
            "novelty",
        ]
        # self.text_preds_dims = {dim: [] for dim in self.dimensions}
        # self.audio_preds_dims = {dim: [] for dim in self.dimensions}
        # self.video_preds_dims = {dim: [] for dim in self.dimensions}
        # self.settings = TinyDB(os.path.join(self.file_dir, "data", "settings.json"), ensure_ascii=False)
        self.get_settings()
        self.turns = []

        def callback():
            pass

        self.call_back_fn = callback
        # self.backendvideo = BackendVideo()

    def get_settings(self):
        self.settings_db = TinyDB(
            os.path.join(self.file_dir, "data", "settings.json"), ensure_ascii=False
        )
        self.settings = self.settings_db.all()[0]
        self.settings_vad = self.settings["VAD"]
        self.logs_dir = os.path.join(self.file_dir, "logs")
        self.session_id = self.settings["session_id"]
        self.db_dir = os.path.join(self.logs_dir, str(self.session_id))
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self.device = self.settings["device"]
        self.label_models = {}
        self.audio_feat_title = self.settings["features"]["audio"]
        self.text_feat_title = self.settings["features"]["text"]
        self.video_feat_title = self.settings["features"]["video"]
        self.logger = self.settings["logger"]  # False # read from settings
        self.get_dbs()

    def get_dbs(self):
        self.db_audio = TinyDB(
            os.path.join(self.db_dir, "backend_audio.json"), ensure_ascii=False
        )
        self.db_main = TinyDB(
            os.path.join(self.db_dir, "backend_main.json"), ensure_ascii=False
        )
        self.db_video = TinyDB(
            os.path.join(self.db_dir, "backend_video.json"), ensure_ascii=False
        )

    def register_audio_events(self):
        for event in self.db_audio.all():
            turn = event["turn"]
            print("turn", turn)
            turn_main = self.db_main.search(Query().turn == turn)
            turn_audio = self.db_audio.search(Query().turn == turn)
            if len(turn_main) != 0:
                print("turn_audio exists", turn_main[0]["turn"])
                pass
            else:
                print(f" {turn} does not exist")
                time_start, time_end = (
                    turn_audio[0]["time_start"],
                    turn_audio[0]["time_end"],
                )
                video_data = self.get_related_video_outputs(time_start, time_end)
                if len(video_data) != 0:
                    audio_turn_data = self.get_turn_info(turn_audio[0], video_data)
                    self.db_main.insert(audio_turn_data)

            # self.turn
        # self.db_main.insert(data)
        self.call_back_fn()

    def get_related_video_outputs(self, time_start, time_end):
        video_data = self.db_video.search(
            (Query().time >= time_start - 1) & (Query().time <= time_end)
        )

        return video_data

    def classify_preds(self, preds, feat_tiltes):
        # preds_all = []
        # print("preds_dicts", preds_dicts)
        # for preds_dict in preds_dicts:
        #     preds_all.append(np.array(list(preds_dict.values())))
        # preds = np.concatenate(preds_all, 1)
        preds_all = []
        for pred in preds:
            preds_all.append(np.array(list(pred)))
        preds = np.concatenate(preds_all, 1)
        # print("preds.shape", preds.shape)
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
        probs_all = {}
        for prob, thr, label in zip(probs, thresholds, self.affect_labels):
            probs_all[label] = prob
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
            influences = [round(inf, 3) for inf in influences]
            classes[f"{label}_influence"] = list(influences)
        # print(classes)
        return classes, probs_all, thresholds

    def get_video_preds(self, video_data):
        # IMPORTANT: next line should be changed to average not -1
        video_preds = [video_data[-1][label] for label in self.affect_labels]
        video_preds_dims = [video_data[-1][label] for label in self.dimensions]
        # face_path = ""
        face_paths = []
        valences = []
        for video_datum in video_data:
            valence = video_datum["intrinsic_pleasantness"]
            frame_path = video_datum["frame_path"]
            if frame_path != "":
                # face_path = frame_path
                valences.append(valence)
                face_paths.append(frame_path)
        try:
            idx = np.argmax(np.abs(valences))
            face_path = face_paths[idx]
        except:
            face_path = ""
        # print("face_path", face_path)
        return video_preds, video_preds_dims, face_path

    def get_turn_info(self, turn_audio, video_data):
        audio_preds = [turn_audio["audio_preds"][label] for label in self.affect_labels]
        text_preds = [turn_audio["text_preds"][label] for label in self.affect_labels]
        # frame_ids = random.sample(video_data["frame_path"],2)
        video_preds, video_preds_dims, face_path = self.get_video_preds(video_data)
        audio_cls, _, _ = self.classify_preds([audio_preds], [self.audio_feat_title])
        # print("audio_cls", audio_cls)
        text_cls, _, _ = self.classify_preds([text_preds], [self.text_feat_title])
        # print("text_cls", text_cls)
        text_audio_cls, _, _ = self.classify_preds(
            [text_preds, audio_preds], [self.text_feat_title, self.audio_feat_title]
        )
        # print("text_audio_cls", text_audio_cls)
        text_audio_video_cls, probs_all, thresholds = self.classify_preds(
            [text_preds, audio_preds, video_preds],
            [self.text_feat_title, self.audio_feat_title, self.video_feat_title],
        )
        print("text_audio_video_cls", text_audio_video_cls)
        print("probs_all", probs_all)
        print("thresholds", thresholds)
        label_details = {}
        label_main_list = []
        label_main_inf = {}
        label_pos_list = []
        label_neg_list = []
        # for cls_k, cls_v in turn_audio["text_audio_cls"].items():
        #     if "influence" in cls_k:
        #         label_main_inf[cls_k.replace("_influence", "")] = cls_v
        #     elif cls_v > 0:
        #         label_main_list.append(cls_k)
        for cls_k, cls_v in text_audio_video_cls.items():
            if "influence" in cls_k:
                label_main_inf[cls_k.replace("_influence", "")] = cls_v
            elif cls_v > 0:
                label_main_list.append(cls_k)
        for label, pred in probs_all.items():
            pred = max(int(np.mean(pred)), 0)
            pred = min(int(np.mean(pred)), 100)
            label_details[label] = pred
            lbl = label + ": " + str(pred)
            if label in self.affect_pos:
                label_pos_list.append(lbl)
            if label in self.affect_neg:
                label_neg_list.append(lbl)
        dim_main_list = []
        dim_details = {}
        dim_detail_list = []
        for text_pred, audio_pred, video_pred, dim in zip(
            turn_audio["text_preds_dims"],
            turn_audio["audio_preds_dims"],
            video_preds_dims,
            self.dimensions,
        ):
            # self.text_preds_dims[dim].append(text_pred)
            # self.audio_preds_dims[dim].append(audio_pred)
            # self.video_preds_dims[dim].append(video_pred)
            # try:  # in case if std is zero
            #     text_pred = (text_pred - np.mean(self.text_preds_dims[dim])) / np.std(
            #         self.text_preds_dims[dim]
            #     )
            #     audio_pred = (
            #         audio_pred - np.mean(self.audio_preds_dims[dim])
            #     ) / np.std(self.audio_preds_dims[dim])
            #     video_pred = (
            #         video_pred - np.mean(self.video_preds_dims[dim])
            #     ) / np.std(self.video_preds_dims[dim])
            # except:
            #     pass
            pred = (1 * text_pred + 1 * audio_pred + 1 * video_pred) / 3
            dim_details[dim] = round(pred, 3)
            dim_detail_list.append(dim + ": " + str(round(pred, 3)))
            if pred > 0.5:
                dim_main_list.append(dim)
        labels_infs = {}
        labels_infs_pos = []
        labels_infs_neg = []
        for emo, infs in label_main_inf.items():
            labels_infs[emo] = infs
            infs_str = [str(round(inf * 100)) + "%" for inf in infs]
            infs_str = ", ".join(infs_str)
            if emo in self.affect_pos:
                labels_infs_pos.append(emo + ": " + infs_str)
            if emo in self.affect_neg:
                labels_infs_neg.append(emo + ": " + infs_str)

        labels_thresholds = {
            label: round(thr, 1) for label, thr in zip(self.affect_labels, thresholds)
        }
        print(labels_thresholds)
        data = {
            "turn": turn_audio["turn"],
            "trs": turn_audio["trs"],
            # "frame_path": str(video_data['frame_path']),
            "time_start": datetime.fromtimestamp(turn_audio["time_start"]).strftime(
                "%d-%m-%Y, %H:%M:%S.%f"
            ),
            "time_end": datetime.fromtimestamp(turn_audio["time_end"]).strftime(
                "%d-%m-%Y, %H:%M:%S.%f"
            ),
            # "audio_preds": turn_audio["audio_preds"],
            "label_main": ", ".join(label_main_list),
            "label_pos": "\n".join(label_pos_list),
            "label_neg": "\n".join(label_neg_list),
            "label_details": label_details,
            "labels_infs": labels_infs,
            "labels_thresholds": labels_thresholds,
            "label_inf_pos": "\n".join(labels_infs_pos),
            "label_inf_neg": "\n".join(labels_infs_neg),
            "dim_main": ", ".join(dim_main_list),
            "dim_detail": "\n".join(dim_detail_list),
            "dim_details": dim_details,
            "face_path": face_path,
        }
        print("data:", data)

        return data


if __name__ == "__main__":
    from settings import Settings

    settings = Settings(session_id=0)
    settings.setup()
    backend = Backend_main()
    backend.register_audio_events()
