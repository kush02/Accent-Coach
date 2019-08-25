
import numpy as np
import simpleaudio as sa
import librosa, librosa.display
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy

from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames, delta_features
from nnmnkwii.util import apply_each2d_trim
from nnmnkwii.metrics import melcd
from nnmnkwii.baseline.gmm import MLPG
import pyworld
import pysptk
from pysptk.synthesis import MLSADF, Synthesizer
from sklearn.mixture import GaussianMixture



def read_text_data(name):
	"""
		Reads a textfile containing audio data
	"""
	
	audio = []
	with open(name,'r') as f:
		data = f.readlines()
		for i in data:
			audio.append(float(i))
	
	return np.array(audio)


def read_wavfile_data(name, sr=16000, resample_type='kaiser_best'):
	"""
		Reads a wavfile or mp3 file containing audio data
	"""

	audio, sr = librosa.load(name,sr=sr,res_type=resample_type)
	
	return audio


def playback_audio(signal,audio_channels=1, bytes_per_sample=2,sr=16000):
	"""
		Play the input waveform
	"""
	
	if type(signal) != 'numpy.ndarray':
		signal = np.array(signal)

	signal *= 32767 / max(abs(signal)) # normalize the signal
	signal = signal.astype(np.int16) # 
	play_obj = sa.play_buffer(signal, audio_channels, bytes_per_sample, sr)
	play_obj.wait_done()

	return


def get_spectrogram(signal, fs=16000, frame_period=5):
	"""
		Extracts spectrogram from signal
	"""

	if type(signal) != 'numpy.float64':
		signal = np.float64(signal)

	f0, timeaxis = pyworld.dio(signal, fs, frame_period=frame_period)
	f0 = pyworld.stonemask(signal, f0, timeaxis, fs)
	spectrogram = pyworld.cheaptrick(signal, f0, timeaxis, fs)	

	return spectrogram


def get_aperiodicity(signal, fs=16000, frame_period=5):
	"""
		Extract aperiodicity of a signal
	"""

	if type(signal) != 'numpy.float64':
		signal = np.float64(signal)

	f0, timeaxis = pyworld.dio(signal, fs, frame_period=frame_period)
	f0 = pyworld.stonemask(signal, f0, timeaxis, fs)
	aperiodicity = pyworld.d4c(signal, f0, timeaxis, fs)

	return aperiodicity


def remove_zero_frames_spectrogram(spectrogram):
	""" 
		Removes frames containing only zeros from spectrogram
	"""

	return trim_zeros_frames(spectrogram)


def get_mfcc(spectrogram, order=12, alpha=1):
	"""
		Extracts MFCCs from spectrogram
	"""

	return pysptk.sp2mc(spectrogram, order=order, alpha=alpha)


def add_delta_features(mfcc, order=2):
	"""
		Extracts the 1st or 2nd order delta features from MFCCs
	"""

	windows = []
	if order == 1:
		windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
    ]

	else:
		windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]

	return apply_each2d_trim(delta_features, mfcc, windows)


def dtw_alignment(x, y, dist=melcd, radius=1, verbose=0):
	"""
		Use Dynamic Time Warping to align MFCCs by minimizing the distance metric between a pair of MFCCs
	"""

	return DTWAligner(verbose=verbose, dist=dist, radius=radius).transform((x, y))


def parameter_generation_MLPG(gmm, added_delta=True, order=2):
	"""
		Parameters generation for transforming MFCCS
	"""

	windows = [
        (0, 0, np.array([1.0])),
    ]

	if added_delta:
		if order == 1:
			windows = [
	        (0, 0, np.array([1.0])),
	        (1, 1, np.array([-0.5, 0.0, 0.5])),
	    ]

		else:
			windows = [
	        (0, 0, np.array([1.0])),
	        (1, 1, np.array([-0.5, 0.0, 0.5])),
	        (1, 1, np.array([1.0, -2.0, 1.0])),
	    ]

	return MLPG(gmm, windows=windows)


def reconstruct_waveform(signal, spectrogram, aperiodicity, fs=16000, frame_period=5):
	"""
		Reconstructs the waveform from the spectrogram and MFCCs
	"""

	if type(signal) != 'numpy.float64':
		signal = np.float64(signal)

	f0, timeaxis = pyworld.dio(signal, fs, frame_period=frame_period)
	f0 = pyworld.stonemask(signal, f0, timeaxis, fs)

	return pyworld.synthesize(f0, spectrogram, aperiodicity, fs, frame_period)





