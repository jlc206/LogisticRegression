import numpy as np
import matplotlib.pyplot as plt
import partition_data as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve, auc
from scipy import interp

logclassifier = LogisticRegression()
linclassifier = LinearRegression()

method = 'lin'

hsp_file = 'SCOREDATA.D3R'
fulldata_file = 'SCOREDATA.vina.balanced'

hsp = pd.create_dict(hsp_file)
data = pd.create_dict(fulldata_file)


######## TRAIN ON WHOLE DATASET, PREDICT FOR HSP90 ########
train_x = []
train_y = []

targets = list(data.keys())
for target in targets:
    targetdata_x = data[target]['x']
    targetdata_y = data[target]['y']
    for features in targetdata_x:
        train_x.append(features)
    for val in targetdata_y:
        train_y.append(val)

test_x = []
test_y = []

targetdata_x = hsp['hsp90']['x']
targetdata_y = hsp['hsp90']['y']
for features in targetdata_x:
    test_x.append(features)
for val in targetdata_y:
    test_y.append(val)
    
if method == 'lin':
    probs_full = linclassifier.fit(train_x, train_y).predict(test_x)
if method == 'log':
    probs_full = logclassifier.fit(train_x, train_y).predict_proba(test_x)
    
print "TRAIN ON WHOLE DATASET"
full = zip(test_y, probs_full.tolist())
if method == 'lin':
    with open('d3r_predictions/alldata_linreg.txt', 'w') as f:
        for line in full:
            f.write(str(line)+'\n')
if method == 'log':
    with open('d3r_predictions/alldata_logreg.txt', 'w') as f:
        for line in full:
            f.write(str(line)+'\n')

######## TRAIN ON HS90A, PREDICT FOR HSP90 ########
train_x = []
train_y = []

targetdata_x = data['hs90a']['x']
targetdata_y = data['hs90a']['y']
for features in targetdata_x:
    train_x.append(features)
for val in targetdata_y:
    train_y.append(val)

test_x = []
test_y = []

targetdata_x = hsp['hsp90']['x']
targetdata_y = hsp['hsp90']['y']
for features in targetdata_x:
    test_x.append(features)
for val in targetdata_y:
    test_y.append(val)
    
if method == 'lin':
    probs_singletarget = linclassifier.fit(train_x, train_y).predict(test_x)
if method == 'log':
    probs_singletarget = logclassifier.fit(train_x, train_y).predict_proba(test_x)
    
print "TRAIN ONLY ON HS90A TARGET"
full = zip(test_y, probs_singletarget.tolist())
if method == 'lin':
    with open('d3r_predictions/singletarget_linreg.txt', 'w') as f:
        for line in full:
            f.write(str(line)+'\n')
if method == 'log':
    with open('d3r_predictions/singletarget_logreg.txt', 'w') as f:
        for line in full:
            f.write(str(line)+'\n')
