import numpy as np
import pysptk as sptk
from scipy.io import wavfile
import os
from pysptk.synthesis import MLSADF, Synthesizer
import pyworld as pw


fs = 16000
fftlen = 512
alpha = 0.42
dim = 25

datalist = []
with open("conf/eval.list", "r") as f:
    for line in f:
        line = line.rstrip()
        datalist.append(line)

for i in range(0,len(datalist)):
    outfile = "result/wav/{}_diff.wav".format(datalist[i])
    with open("data/SF-TF/mgc/{}.mgc".format(datalist[i]), "rb") as f:
        conv_mgc = np.fromfile(f, dtype="<f8", sep="")
        conv_mgc = conv_mgc.reshape(len(conv_mgc)//dim, dim)

    with open("data/SF/mgc/{}.mgc".format(datalist[i]), "rb") as f:
        src_mgc = np.fromfile(f, dtype="<f8", sep="")
        src_mgc = src_mgc.reshape(len(src_mgc)//dim, dim)
    
    with open("data/SF-TF/f0/{}.f0".format(datalist[i]),"rb") as f:
        f0 = np.fromfile(f, dtype="<f8", sep="")
        
    with open("data/SF/ap/{}.ap".format(datalist[i]),"rb") as f:
        ap = np.fromfile(f, dtype="<f8", sep="")
        ap = ap.reshape(len(ap)//(fftlen+1),fftlen+1)

    # 先にピッチ変換する
    fs, _ = wavfile.read("data/SF/wav/{}.wav".format(datalist[i]))
    sp = sptk.mc2sp(src_mgc.astype(np.float64), alpha, fftlen*2)
    owav = pw.synthesize(f0, sp, ap, fs)
    owav = np.clip(owav, -32768, 32767)
    data = owav

    # その後、差分スペクトル法を適用
    diff_mgc = conv_mgc - src_mgc  # 差分のフィルタを用意する
    diff_mgc = np.zeros(shape=conv_mgc.shape)
    b = np.apply_along_axis(sptk.mc2b, 1, diff_mgc, alpha)
    synthesizer = Synthesizer(MLSADF(order=dim-1, alpha=alpha), 80)
    owav = synthesizer.synthesis(data, b)
    
    owav = np.clip(owav, -32768, 32767)
    wavfile.write(outfile, fs, owav.astype(np.int16))
