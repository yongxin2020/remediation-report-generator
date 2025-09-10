import os
from tinydb import TinyDB, Query

class Settings():
    def __init__(self, session_id = 0):
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(self.file_dir, "data")
        self.db = TinyDB(os.path.join(data_path, "settings.json"), ensure_ascii=False)
        self.session_id = session_id
    
    def init_data(self):
        self.data = {
                    "session_id": self.session_id,
                    "device": "cpu",
                    "logger": True,
                    "save_wav": True,
                    "front_end":
                        {
                            "theme": "Light Gray 1",
                            "font_type": "Verdana",
                            "font_size": 16,
                        },
                    "features":
                        {
                            "text": "tfidf_google", 
                            "audio": "mfb_normed", 
                            "video": "clip"
                        },
                    "VAD":
                        {
                            "listen":       True,
                            "listen_time":  60*60, # 1h
                            "chunk_sec":    .8,
                            "buffer_sec":   15,
                            "frame_rate":   16000,
                            "hys_top":      0.5,
                            "hys_bottom":   0.0,
                            "cutWin_sec":   0.1,
                            "mergeWin_sec": 0.7,
                        }
                    }

    def setup(self):
        self.init_data()
        query = Query()
        if self.db.search(query.features == self.data["features"]):#self.db.get(query.features) is None:
            self.db.update(self.data)
        else:
            self.db.insert(self.data)

if __name__== "__main__":
    settings = Settings(session_id = 0)
    settings.setup()
    print("settings tests passed")