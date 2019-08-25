import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import librosa
import pickle
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras import metrics, regularizers
from keras.utils import to_categorical

import AccentClassifier as ac


######################################################## GETTING AND DUMPING THE DATA 
data = pd.read_csv('speakers_all.csv')
c = Counter(data['native_language'])
#print(c.most_common()) # use english, spanish
"""
folder = 'use/'
record = ac.extract_data(folder)

with open('record.pickle','wb') as f:
	pickle.dump(record,f)

print('done')
"""

######################################################## ADD NOISE + EFFECTS TO EACH RECORDING FOR DATA AUGMENTATION
"""
record = {}
with open('record.pickle','rb') as f:
	record = pickle.load(f)

X_noise, y = [], []
for key in record.keys():
	target = ''
	if 'mandarin' in key:
		target = 'mand'
	elif 'english' in key:
		place = data[data['filename']==key]['country'].values
		if ('usa' in place or 'canada' in place):
			target = 'eng_us_canada'
		elif ('new zealand' in place or 'australia' in place) :
			target = 'eng_aus_nz'
		elif ('uk' in place or 'ireland' in place):
			target = 'eng_uk'
		else:
			continue
	else:
		continue		

	print(key)
	signal = record[key]
	X, Y = ac.data_augmentation(signal, target)
	X_noise.append(X)
	y.append(Y)
	


X_noise = list(itertools.chain.from_iterable(X_noise))
y = list(itertools.chain.from_iterable(y))

print(len(X_noise),len(y))
print(Counter(y))
 
######################################################## EXTRACTING MFCCs

num_sec = 5 
sr = 16000
X = ac.create_mfcc_matrix(X_noise,num_sec=num_sec,sr=sr,highfreq=8000)
print(X.shape)

######################################################### HANDLING CLASS IMBALANCE WITH OVERSAMPLING

print('start oversampling')
X,y = ac.oversample(X,y,random_state=42)
print('done oversampling')
print(X.shape)
print(Counter(y))
print(len(y))
"""
######################################################### SAVE MFCCs TO FILE

#ac.save_file('X.npy',X)
#ac.save_file('y.npy',y)
#X = ac.load_file('X.npy')
#y = ac.load_file('y.npy')
print('done loading')
print(X.shape,y.shape)

######################################################### WRITING MFCCs and LABELS TO TEXT FILE

#ac.write_to_file('mfcc_data5secs.txt',X,delimiter='\t',fmt='%f')
#ac.write_to_file('labels5secs.txt',y,fmt='%s',delimiter='\n')
print('done making text file')


######################################################### TRAINING CLASSIFIER


row,col = X[0].shape #498,26

n_classes = len(np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

enc = LabelEncoder()
y_train = enc.fit_transform(y_train)
y_test = enc.transform(y_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

batch_size = 128
nb_epoch = 10

print("Starting ConvNet")
model = ac.base_convnet(row,col,n_classes=n_classes)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[metrics.categorical_accuracy])
model.fit(X_train, y_train, batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_split=0.2)

y_class_prob = model.predict(X_test)
y_pred = np.around(y_class_prob)

score = model.evaluate(X_test, y_test,  verbose=1)
print(model.metrics_names)
print(score)
ac.print_classifier_metrics(y_test,y_pred,name="ConvNet",average='weighted')

lab_pred = np.argmax(y_pred, axis=1)
lab_pred = enc.inverse_transform(lab_pred)
lab_true = np.argmax(y_test, axis=1)
lab_true = enc.inverse_transform(lab_true)

class_names = enc.classes_
plt.figure()
cm = confusion_matrix(lab_true,lab_pred)
ac.plot_confusion_matrix(cm, classes=class_names, title='ConvNet')
plt.show()



