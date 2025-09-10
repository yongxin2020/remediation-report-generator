import cv2, os, time, datetime
from datetime import datetime
import numpy as np
import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from face_detection import RetinaFace
from tinydb import TinyDB, Query
from settings import Settings
from threading import Thread
import pytz  


class BackendVideo:
    def __init__(self):
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.file_dir, "data")
        self.models_dir = os.path.join(self.file_dir, "models")
        self.recording = True
        # self.out_file = TinyDB(os.path.join(self.file_dir, "data", "preds.json"))

        self.dimensions = [
            "arousal",
            "coping",
            "goal_conduciveness",
            "intrinsic_pleasantness",
            "novelty",
        ]

        self.affect_labels = ["annoyed", "anxious", "confident", "desperate", "frustrated", 
                              "happy", "interested", "relaxed", "satisfied", "surprised"]


        self.get_settings_from_file()
        self.init_models()

        logs_dir = os.path.join(self.file_dir, "logs")
        self.db_dir = os.path.join(logs_dir, str(self.session_id))
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        if self.logger:
            self.out_file = TinyDB(
                os.path.join(self.db_dir, "backend_video.json"), ensure_ascii=False
            )
        self.frame_dir = os.path.join(logs_dir, str(self.session_id),'frames')
        if not os.path.exists(self.frame_dir):
            os.makedirs(self.frame_dir)

    def get_settings_from_file(self):
        settings_db = TinyDB(
            os.path.join(self.file_dir, "data", "settings.json"), ensure_ascii=False
        )
        self.settings = settings_db.all()[0]
        self.feature_title = self.settings["features"]["video"]
        self.device = self.settings["device"]
        self.session_id = self.settings["session_id"]
        self.logger = self.settings["logger"]
        self.feature_title = "clip"

    def returnCameraIndexes(self):
        # checks the first 10 indexes.
        index = 0
        arr = []
        i = 10
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        return arr[-1]

    def init_models(self):
        if self.feature_title == "clip":
            self.init_clip_model()
        elif self.feature_title == "fau":
            self.init_fau_model()
        else:
            raise ValueError("Invalid feature title: {}".format(self.feature_title))
        model_path = os.path.join(self.models_dir, self.feature_title, "model_s.pt")
        self.model_dim_sum = torch.jit.load(model_path, map_location=self.device).to(
            self.device
        )
        model_path = os.path.join(self.models_dir, self.feature_title, "model_c.pt")
        self.model_dim_con = torch.jit.load(model_path, map_location=self.device).to(
            self.device
        )
        self.label_models = []
        for label in self.affect_labels:
            path = os.path.join(self.models_dir, self.feature_title, f"{label}.pth")
            model = torch.jit.load(path, map_location=self.device)
            self.label_models.append(model)
        # self.affect_labels

    def init_clip_model(self):
        self.feat_processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def get_features(self, frame,frame_id,datetime_str):
        if self.feature_title == "clip":
            image = self.detect_face_retina(frame)

            inputs = self.feat_processor(images=image, return_tensors="pt").to(
                self.device
            )
            outputs = self.clip(**inputs)
            image_embeds = outputs.image_embeds
            feats = image_embeds.squeeze().detach().cpu().numpy()


            if frame_id % 30 == 0:
                frame_path = os.path.join(self.frame_dir,"frame_"+ datetime_str+".jpg")
                cv2.imwrite(frame_path, image)

                return feats, frame_path
            else: return feats, ""



        # elif self.feature_title == "fau":
        # Extract FAU features
        else:
            raise ValueError("Invalid feature title: {}".format(self.feature_title))

    def get_predictions(self, buffer):
        out_sum = self.model_dim_sum(
            torch.tensor(buffer)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
        )
        out_sum = out_sum.detach().cpu().numpy().squeeze()

        out_con = self.model_dim_con(
            torch.tensor(buffer)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
        )
        out_con = out_con.detach().cpu().numpy().squeeze()

        out_lbls = []
        for l, label in enumerate(self.affect_labels):
            out = self.label_models[l](
                torch.tensor(buffer)
                .to(device=self.device, dtype=torch.float32)
                .unsqueeze(0)
            )
            out = out.detach().cpu().numpy().squeeze()
            out_lbls.append(out)
            
        out_sum_norm = abs(out_sum - np.min(out_sum) / (np.max(out_sum) - np.min(out_sum)))
        out_con_norm = abs((out_con - np.min(out_con, axis=0)) / (np.max(out_con, axis=0) - np.min(out_con, axis=0)))
        return out_sum, out_con, out_lbls

    def save_output(self, t1, t2, pred_dim, pred_lbls,frame_path):
        pred_dim = np.mean(pred_dim, axis=0)
        print(pred_dim)
        dic = {
            "datetime": str(t1),
            "time": float(t2),
            "frame_path": frame_path,
        }
        for d, dim in enumerate(self.dimensions):
            dic[dim] = round(float(pred_dim[d]), 3)
        for l, label in enumerate(self.affect_labels):
            preds = [int(100 * round(float(p), 3)) for p in pred_lbls[l]]
            # for p, pred in enumerate(preds):
            #     if pred < 0:
            #         preds[p] = 0
            dic[label] = preds
        self.out_file.insert(dic)

    def detect_face_retina(self, img):
        try:
            detector = RetinaFace()
            faces = detector(img)
            if faces is None:
                raise Exception("No face found")
            else:
                box, _, _ = faces[0]
                box = np.asarray(box, dtype="int")
                x, y, w, h = box[0], box[1], box[2], box[3]
                roi = img[y:h, x:w]

                new_width = 112
                aspect_ratio = float(new_width) / roi.shape[1]
                new_height = int(roi.shape[0] * aspect_ratio)
                frame_resized = cv2.resize(roi, (new_width, new_height))

                return frame_resized
        
        except Exception as e:
            print(f"Error: {e}")
            return None  

    def detect_face_retina2(self, img):
        try:
            detector = RetinaFace()
            faces = detector(img)
            if faces is None or len(faces) == 0:
                print("No face found")
                return None 
            
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            box, _, _ = largest_face
            box = np.asarray(box, dtype="int")
            x, y, w, h = box[0], box[1], box[2], box[3]
            roi = img[y:h, x:w]

            new_width = 112
            aspect_ratio = float(new_width) / roi.shape[1]
            new_height = int(roi.shape[0] * aspect_ratio)
            frame_resized = cv2.resize(roi, (new_width, new_height))

            return frame_resized
        except Exception as e:
            print(f"Error: {e}")
            return None


    def process_video_stream_run(self):
        cam_idx = self.returnCameraIndexes()
        cap = cv2.VideoCapture(cam_idx)
        buffer = []
        MAX_BUFFER_SIZE = 10
        frame_id = 0

        while self.recording:
            print(self.recording)
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if not ret:
                print("Error reading frame from camera")
                break

            frame_id += 1

            if frame_id % 10 == 0:
                current_time = time.time()
                #datetime_str = datetime.fromtimestamp(current_time).strftime(
                #    "%Y-%m-%d %H:%M:%S.%f"
                #)

                france_timezone = pytz.timezone('Europe/Paris')

                datetime_obj = datetime.fromtimestamp(current_time, tz=france_timezone)
                datetime_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")
                try:
                    feats,frame_path = self.get_features(frame,frame_id,datetime_str)
                    buffer.append(feats)
                    
                    if len(buffer) > MAX_BUFFER_SIZE:
                        buffer.pop(0) 

                    if len(buffer) > 3:
                        out_sum, out_con, out_lbls = self.get_predictions(buffer)
                        self.save_output(datetime_str, current_time, out_con, out_lbls,frame_path)
                except Exception as E:
                    print(E)

        cap.release()
        cv2.destroyAllWindows()

    def stop_recording(self):
        self.recording = False

    def process_video_stream(self):
        T = Thread(target=self.process_video_stream_run, args=())
        T.start()


if __name__ == "__main__":
    backend = BackendVideo()
    backend.process_video_stream()
