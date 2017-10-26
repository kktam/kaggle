# Binary Classification - Kaggle Titanic
# https://www.kaggle.com/c/titanic

import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("train.csv", sep=',', header=0)

dataset = dataframe
# split into input (X) and output (Y) variables
X = dataset.iloc[:, np.r_[0, 2:11, 11]]
X['FSize'] = X['SibSp'] + X['Parch']
Y = dataset[dataset.columns[1]]

#print(X)
#print(Y)
#exit()

X = X['FSize'][0:890].values
Y = Y[0:890].values

print(X)
#print(Y)
#exit()

# encode class values as integers
#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)

# random forest model fitting
#estimator = RandomForestClassifier(n_estimators=10)
#estimator = estimator.fit(X, Y)
#results = cross_val_score(estimator, X, Y)
#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#exit()

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(800, input_dim=800, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
	
# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

