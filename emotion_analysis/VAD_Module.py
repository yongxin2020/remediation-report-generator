import numpy as np
import torch

class VAD_Module(object):
    """
    VAD module to predict voice probabilities of frames and get speech segments.
        
    """
    def __init__(self, device="cpu", model_path=""):
        self.device = device
        self.model_path = model_path
        # self.model = torch.load(model_path, map_location=device)
        self.model = torch.jit.load(model_path, map_location=device)

    def predict(self, feats):
        self.outs = self.model(feats).squeeze().squeeze().cpu().detach().numpy()
        return self.outs

    def post_process(self, feats_fs=25, hys_top=0.0, hys_bottom=0.0, cutWin_sec=0.1, mergeWin_sec=0.7):
        out = self.hysteresis(self.outs, bottom=hys_bottom, top=hys_top)# Smoothing the output of the model with a window of 0.5s (here it means 50 samples based on feature extraction process)
        out = self.ignore_shorts(out, win=cutWin_sec/feats_fs)
        out = self.mergeOuts(out, win=mergeWin_sec/feats_fs)
        self.outs_pp = out
        self.times = self.getTimes(out)
        return self.times

    def hysteresis(self, sig, bottom=-0.8, top=0.75):
        mysig = sig.copy()
        mysig[0] = 1 if mysig[0] > top else -1
        for i in range(1, len(mysig)):
            if mysig[i] >= top:
                mysig[i] = 1
            if mysig[i] >= bottom and mysig[i] < top:
                if mysig[i-1] == 1: 
                    mysig[i] = 1
                else:
                    mysig[i] = -1
            if mysig[i] < bottom:
                mysig[i] = -1
        return mysig

    def ignore_shorts(self, sig, win=25):
        mysig = sig.copy()
        start = 0
        for i in range(1, len(mysig)):
            if mysig[i] == 1 and mysig[i-1] == -1: start = i
            if mysig[i] == -1 and mysig[i-1] == 1: 
                if i-start < win: 
                    for j in range(start, i):
                        mysig[j] = -1 
        return mysig

    def mergeOuts(self, out, win=25):
        myOut = out.copy()
        counter = 0; shouldCount = False; startC = 0
        for i in range(1, len(out)):
            if out[i-1] == 1 and out[i] == -1: 
                shouldCount = True
                startC = i
            if shouldCount:
                counter += 1
            if out[i-1] == -1 and out[i] == 1: 
                shouldCount = False
                if counter < win:
                    for j in range(startC, i):
                        myOut[j] = 1
                counter = 0
            if out[i-1] == -1 and out[i] == -1:
                startC = i
        return myOut

    def getTimes(self, out):
        # if "W2V2" in self.feat or "wav2vec2" in self.feat: fs=0.02
        ins = []
        outs = []
        last = 0
        for i, o in enumerate(out):
            if o == 1 and last != 1: ins.append(i)
            if o == -1 and last == 1: outs.append(i)
            last = o
        if out[-1] == 1: outs.append(len(out)-1)
        times = []
        length = len(out)
        for i, _ in enumerate(outs):
            times.append([round(ins[i]/length,3), round(outs[i]/length,3)])
        self.times = times
        return times
        