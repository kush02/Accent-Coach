
import numpy as np
import simpleaudio as sa
import librosa, librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import pandas as pd
import pickle

from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.metrics import melcd
from nnmnkwii.baseline.gmm import MLPG
import pyworld
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer
from sklearn.mixture import GaussianMixture

import VoiceConversion as vc



file = 'english50.mp3'
audio_eng = vc.read_wavfile_data(file)
#audio_eng = audio_eng[:sr*20]
print(audio_eng.shape)


file = 'mandarin50.mp3'
audio_mand = vc.read_wavfile_data(file)
#audio_mand = audio_mand[:sr*20]
print(audio_mand.shape)


fs = 16000
fftlen = pyworld.get_cheaptrick_fft_size(fs)
alpha = pysptk.util.mcepalpha(fs)
order = 24
frame_period = 5
hop_length = int(fs * (frame_period * 0.001))
use_delta = True

if use_delta:
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
else:
    windows = [
        (0, 0, np.array([1.0])),
    ]


record = {}
with open('record.pickle','rb') as f:
	record = pickle.load(f)

data_count = 5
mc_mand = []
mand_count = 0
for key in record.keys():
	if mand_count == data_count:
		break
	elif 'mandarin' in key:
		mand_count += 1
		audio = record[key]
		spec = vc.get_spectrogram(audio,fs=fs,frame_period=frame_period)
		mc = vc.get_mfcc(spec, order=order, alpha=alpha)
		mc_mand.append(mc)	

mc_eng = []
data = pd.read_csv('speakers_all.csv')
eng_count = 0
for key in record.keys():
	if eng_count == data_count:
		break
	elif 'english' in key:
		place = data[data['filename']==key]['country'].values
		if ('usa' in place or 'canada' in place):
			eng_count += 1
			audio = record[key]
			spec = vc.get_spectrogram(audio,fs=fs,frame_period=frame_period)
			mc = vc.get_mfcc(spec, order=order, alpha=alpha)
			mc_eng.append(mc)
			
mc_eng = np.vstack(mc_eng)
mc_mand = np.vstack(mc_mand)
print(mc_eng.shape, mc_mand.shape)


mc_eng_expanded = np.expand_dims(mc_eng,axis=0)
mc_mand_expanded = np.expand_dims(mc_mand,axis=0)
X_aligned, Y_aligned = vc.dtw_alignment(mc_eng_expanded, mc_mand_expanded, radius=50)
print(X_aligned.shape, Y_aligned.shape)
X_aligned, Y_aligned = X_aligned[:, :, 1:], Y_aligned[:, :, 1:]
X_aligned = vc.add_delta_features(X_aligned)  
Y_aligned = vc.add_delta_features(Y_aligned)
XY = np.concatenate((X_aligned, Y_aligned), axis=-1).reshape(-1, X_aligned.shape[-1]*2)
XY = remove_zeros_frames(XY)
print(XY.shape)

gmm = GaussianMixture(n_components=128, covariance_type="full", max_iter=1000, verbose=1,random_state=42)
gmm.fit(XY)
paramgen = vc.parameter_generation_MLPG(gmm)


spectrogram = vc.get_spectrogram(audio_eng,fs=fs,frame_period=frame_period)
aperiodicity = vc.get_aperiodicity(audio_eng)
mc = vc.get_mfcc(spectrogram, order=order, alpha=alpha)
c0, mc = mc[:, 0], mc[:, 1:]
mc = delta_features(mc,windows=windows)
mc = paramgen.transform(mc)
mc = np.hstack((c0[:, None], mc))
print(mc.shape)
spectrogram = pysptk.mc2sp(mc.astype(np.float64), alpha=alpha, fftlen=fftlen)
waveform = vc.reconstruct_waveform(audio_eng, spectrogram, aperiodicity, fs, frame_period)

plt.figure()
plt.plot(audio_mand,'r')
plt.figure()
plt.plot(waveform,'b')
#plt.show()
vc.playback_audio(waveform)



