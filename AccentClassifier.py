import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import pickle
import os
import itertools

from python_speech_features import sigproc, base
from scipy.fftpack import dct
from python_speech_features import delta

from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras import metrics, regularizers


def extract_data(folder, sr=16000, res_type='kaiser_fast'):

	record = {}
	for file in os.listdir(folder):
		y,sr = librosa.load(folder+file,sr=sr,res_type=res_type)
		title = file.split('.')[0]
		record[title] = y

	return record


def data_augmentation(signal, target):
	"""
		Perform 6 different types of data augmentation
	"""

	X_noise, y = [], []

	X_noise.append(signal);	y.append(target)
	#num_5_secs_segments = int(len(signal)/80000)
	#signal = signal[:num_5_secs_segments*80000]
	#for i in np.split(signal,num_5_secs_segments):
		#X_noise.append(signal); y.append(target)

	noise = np.random.normal(0,1,len(signal))
	signal_noisy = 0.01*noise + signal
	X_noise.append(signal_noisy); y.append(target)
	#for i in np.split(signal_noisy,num_5_secs_segments):
		#X_noise.append(signal_noisy); y.append(target)

	noise = np.random.normal(0,1,len(signal))  # add two noisy signals with different strengths for each original signal
	signal_noisy = 0.1*noise + signal
	X_noise.append(signal_noisy); y.append(target)
	#for i in np.split(signal_noisy,num_5_secs_segments):
		#X_noise.append(signal_noisy); y.append(target)

	signal_fast = librosa.effects.time_stretch(signal,1.1)
	X_noise.append(signal_fast); y.append(target)
	#num_5_secs_segments = int(len(signal_fast)/80000)
	#signal_fast = signal_fast[:num_5_secs_segments*80000]
	#for i in np.split(signal_fast,num_5_secs_segments):
		#X_noise.append(signal_fast); y.append(target)

	signal_fast = librosa.effects.time_stretch(signal,1.2)   # add two fast signals with different strengths for each original signal and pad them to make them equal length
	X_noise.append(signal_fast); y.append(target)
	#num_5_secs_segments = int(len(signal_fast)/80000)
	#signal_fast = signal_fast[:num_5_secs_segments*80000]
	#for i in np.split(signal_fast,num_5_secs_segments):
		#X_noise.append(signal_fast); y.append(target)

	signal_slow = librosa.effects.time_stretch(signal,0.95)
	X_noise.append(signal_slow); y.append(target)
	#num_5_secs_segments = int(len(signal_slow)/80000)
	#signal_slow = signal_slow[:num_5_secs_segments*80000]
	#for i in np.split(signal_slow,num_5_secs_segments):
		#X_noise.append(signal_slow); y.append(target)

	signal_slow = librosa.effects.time_stretch(signal,0.85)  # add two slow signals with different strengths for each original signal and pad them to make them equal length
	X_noise.append(signal_slow); y.append(target)
	#num_5_secs_segments = int(len(signal_slow)/80000)
	#signal_slow = signal_slow[:num_5_secs_segments*80000]
	#for i in np.split(signal_slow,num_5_secs_segments):
		#X_noise.append(signal_slow); y.append(target)

	return X_noise, y


def compute_mfcc(signal,sr=16000,winlen=0.032,winstep=0.01,numcep=13, nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,window=np.hamming):
	
	signal = sigproc.preemphasis(signal,preemph)
	frames = sigproc.framesig(signal, winlen*sr, winstep*sr, winfunc=window)
	magspec = np.absolute(np.fft.rfft(frames, nfft))
	powspec = 1.0/nfft * np.square(magspec)
	energy = np.sum(powspec,1) # this stores the total energy in each frame
	energy = np.where(energy == 0,np.finfo(float).eps,energy) # if energy is zero, we get problems with log
	fb = base.get_filterbanks(nfilt,nfft,sr,lowfreq,highfreq)
	feat = np.dot(powspec,fb.T) # compute the filterbank energies
	feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log
	feat = np.log10(feat)
	feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
	feat = base.lifter(feat,ceplifter)
	if appendEnergy: feat[:,0] = np.log10(energy) # replace first cepstral coefficient with log of frame energy

	return feat


def compute_delta(mfccs,win=2):
	
	deltas = delta(mfccs,win) 

	return deltas


def mfcc_normalize(feat,mean_normalize=True,var_normalize=False):

	if mean_normalize: feat -= np.mean(feat,axis=0)+1e-8
	if var_normalize: feat /= np.std(feat,axis=0)+1e-8

	return feat


def create_mfcc_matrix(X_noise, num_sec=5, sr=16000, append_delta=True, normalized=True,winlen=0.032,winstep=0.01,numcep=13, nfilt=26,nfft=512,lowfreq=0,
	highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True,window=np.hamming, delta_win=2):

	X = []
	for i in X_noise:
		signal = i[:int(num_sec*sr)] # keep first n seconds of signal
		mfcc = compute_mfcc(signal,sr=sr,winlen=winlen,winstep=winstep,numcep=numcep, nfilt=nfilt,nfft=nfft,lowfreq=lowfreq,
			highfreq=highfreq,preemph=preemph,ceplifter=ceplifter,appendEnergy=appendEnergy,window=window)
		
		if append_delta:
			delta = compute_delta(mfcc,win=delta_win)
			mfcc = np.concatenate((mfcc,delta),axis=1)
		
		if normalized:
			mfcc = mfcc_normalize(mfcc)
		
		X.append(mfcc)

	X = np.array(X)

	return X


def oversample(X,y,random_state=None):
	"""
		Oversamples the minority classes by making synthetic samples using the SMOTE algorithm

	"""

	row, col = X[0].shape

	X_flatten = X.reshape(X.shape[-3],-1)
	X_sampled, y_sampled = SMOTE(random_state=random_state).fit_resample(X_flatten,y)

	X_new = np.reshape(X_sampled,(X_sampled.shape[0],row,col))
	
	return X_new, y_sampled


def save_file(name,X):
	"""
		Saves X as a numpy file in working directory
	"""

	np.save(name,X)

	return


def load_file(name,encoding='ASCII'):
	"""
		Loads the file from working directory
	"""

	return np.load(name,encoding=encoding)


def write_to_file(X,name,delimiter='\t',fmt='%s'):
	"""
		Write X to text file
	"""

	np.savetxt(name,X,delimiter=delimiter,fmt=fmt)

	return


def print_classifier_metrics(y_test,y_pred,name="",average='binary'):
    print("Accuracy score for %s: %f" %(name,accuracy_score(y_test,y_pred)))
    print("Recall score for %s: %f" % (name,recall_score(y_test,y_pred,average=average)))
    print("Precision score for %s: %f" % (name,precision_score(y_test,y_pred,average=average)))
    print("F-1 score for %s: %f" % (name,f1_score(y_test,y_pred,average=average)))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def base_convnet(row,col,n_classes=1):

	model = Sequential()
	model.add(Convolution1D(32,3,input_shape=(row, col)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size=2,strides=1))
	model.add(Dropout(0.15))
	model.add(Convolution1D(64,3,padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size=2,strides=1))
	model.add(Dropout(0.15))
	model.add(Flatten())
	model.add(Dense(n_classes,activation='softmax'))

	return model



