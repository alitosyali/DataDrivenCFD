import numpy as np 
import pandas as pd 
import time
import os

from sklearn.ensemble import RandomForestRegressor
from rawDataWriter_2v import rawDataWriter_2v

# variables to control
discrapency_of_interest = 'r1_log'
take_log = True

# //////////////////////////////////////////////////////////////////////////
#                               READING THE DATA
# /////////////////////////////////////////////////////////////////////////

print("\nReading the datasets...\n")

# parent directory
parent_directory = os.path.dirname(os.getcwd())
path = parent_directory + '\data\\'

# datasets
features_01 = pd.read_csv(path + 'features_case01.csv')
features_02 = pd.read_csv(path + 'features_case02.csv')
features_03 = pd.read_csv(path + 'features_case03.csv')
features_04 = pd.read_csv(path + 'features_case04.csv')
features_05 = pd.read_csv(path + 'features_case05.csv')
discrapencies_01 = pd.read_csv(path + 'discrapencies_case01.csv')
discrapencies_02 = pd.read_csv(path + 'discrapencies_case02.csv')
discrapencies_03 = pd.read_csv(path + 'discrapencies_case03.csv')
discrapencies_04 = pd.read_csv(path + 'discrapencies_case04.csv')
discrapencies_05 = pd.read_csv(path + 'discrapencies_case05.csv')

discrapency_list = ['r1_log', 'r1']
feature_list = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10']

# //////////////////////////////////////////////////////////////////////////
#                         CONVERTING TO ARRAYS
# /////////////////////////////////////////////////////////////////////////

discrapency_01 = np.array(discrapencies_01[discrapency_of_interest])
discrapency_02 = np.array(discrapencies_02[discrapency_of_interest])
discrapency_03 = np.array(discrapencies_03[discrapency_of_interest])
discrapency_04 = np.array(discrapencies_04[discrapency_of_interest])
discrapency_05 = np.array(discrapencies_05[discrapency_of_interest])

features_01 = np.array(features_01[feature_list])
features_02 = np.array(features_02[feature_list])
features_03 = np.array(features_03[feature_list])
features_04 = np.array(features_04[feature_list])
features_05 = np.array(features_05[feature_list])

# //////////////////////////////////////////////////////////////////////////
#                    SPLITTING INTO TRAINING AND TESTING
# /////////////////////////////////////////////////////////////////////////

train_features = np.concatenate((features_01, features_02, features_04), axis=0)
train_discrapency = np.concatenate((discrapency_01, discrapency_02, discrapency_04), axis=0)
test_features = features_03
test_discrapency = discrapency_03

print("\nTraining features shape: {}".format(train_features.shape))
print("Training labels shape: {}".format(train_discrapency.shape))
print("Testing features shape: {}".format(test_features.shape))
print("Testing labels shape: {}\n".format(test_discrapency.shape))

# //////////////////////////////////////////////////////////////////////////
#                    LEARNING
# /////////////////////////////////////////////////////////////////////////

print("\nFitting the model...\n")
start_time = time.time()
# the model
rf = RandomForestRegressor(
						n_estimators=100,
						min_samples_leaf=2,
						max_features='auto',
						bootstrap=True,
						n_jobs=-1,
						verbose=2,
						random_state=42
						)

# train the model
rf.fit(train_features, train_discrapency.ravel())

print("\nTime for fitting: {} seconds".format(time.time()-start_time))

# feature importance
feat_imp = rf.feature_importances_
print(feat_imp)

# making predictions
start_time = time.time()
print("\nMaking predictions...\n")
preds = rf.predict(test_features)

print("\nTime for predicting: {} seconds".format(time.time()-start_time))

# //////////////////////////////////////////////////////////////////////////
#                    RESULTS
# /////////////////////////////////////////////////////////////////////////

print("\nWriting predictions to a csv file...\n")
df = pd.DataFrame(preds, columns=['predictions'])
df.to_csv(parent_directory + '\\results\\r1Log_Pred_case03.csv')

print("\nWriting the raw data file for predictions...\n")
rawDataWriter_2v(predictions=df, discrapency_name='kMean', case='case_03', take_log=take_log)

# # save the model to disk
# filename = 'learned_model.sav'
# joblib.dump(rf, filename)














