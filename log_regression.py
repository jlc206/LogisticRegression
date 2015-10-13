import numpy as np
import matplotlib.pyplot as plt
import partition_data as pd
import argparse
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve, auc, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from scipy import interp


def ccv_plot_roc(num_folds):
    global data
    folds = pd.create_folds(data, num_folds)
    classifier = LogisticRegression()
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i in range(num_folds):
        test_x, test_y, train_x, train_y = pd.split_into_sets(data, folds, i)
        probs = classifier.fit(train_x, train_y).predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1]) #takes, y_true and y_score
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(folds) 
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%d-fold Clustered Cross-Validation' % num_folds)
    plt.legend(loc="lower right")
    plt.show()

def logreg_precision_recall_ccv(num_folds):

    global data
    folds = pd.create_folds(data, num_folds)
    classifier = LogisticRegression()
    
    mean_recall = 0.0
    mean_precision = 0.0
    for i in range(num_folds):
        test_x, test_y, train_x, train_y = pd.split_into_sets(data, folds, i)
        probs = classifier.fit(train_x, train_y).predict_proba(test_x)
        
        y_pred = [1 if x >= .5 else 0 for x in probs[:, 1]]
        
#         print test_y
#         print y_pred
        
        recall = recall_score(test_y, y_pred) #y_true, y_pred
#         print 'RECALL'
#         print recall
        
        precision = precision_score(test_y, y_pred)
#         print 'PRECISION'
#         print precision
#         
        mean_recall += recall
        mean_precision += precision


    mean_precision /= len(folds)
    mean_recall /= len(folds)
    
    print "MEAN PRECISION"
    print mean_precision
    print "MEAN RECALL"
    print mean_recall

def linreg_ccv_plot_roc(num_folds):

    global data
    folds = pd.create_folds(data, num_folds)
    classifier = LinearRegression()
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i in range(num_folds):
        test_x, test_y, train_x, train_y = pd.split_into_sets(data, folds, i)
        probs = classifier.fit(train_x, train_y).predict(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, probs) #takes, y_true and y_score
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(folds) 
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%d-fold Clustered Cross-Validation' % num_folds)
    plt.legend(loc="lower right")
    plt.show()   
    
def rfc_ccv_plot_roc(num_folds):
    global data
    folds = pd.create_folds(data, num_folds)
    classifier = RandomForestClassifier()
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i in range(num_folds):
        test_x, test_y, train_x, train_y = pd.split_into_sets(data, folds, i)
        probs = classifier.fit(train_x, train_y).predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1]) #takes, y_true and y_score
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(folds) 
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%d-fold Clustered Cross-Validation' % num_folds)
    plt.legend(loc="lower right")
    plt.show()
    
def precision_recall_curve(num_folds): #w ccv 10fold
    #haven't tested that this works yet
    global data
    
    folds = pd.create_folds(data, num_folds)
    classifier = LogisticRegression()
    
    for j in range(num_folds):
        test_x, test_y, train_x, train_y = pd.split_into_sets(data, folds, j)
        probs = classifier.fit(train_x, train_y).predict_proba(test_x)
        
        precision, recall, _ = precision_recall_curve(test_y, probs[:, 1])
        print precision
        print recall
               
        precision = dict()
        recall = dict()
        average_precision = dict()
        #for i in range(n_classes):
        for i in range (2): #2 classes?
            precision[i], recall[i], _ = precision_recall_curve(test_y, probs[:, 1])
            average_precision[i] = average_precision_score(test_y, probs[:, 1])

        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = precision_recall_curve(test_y.ravel(), probs[:, 1].ravel())
        average_precision["micro"] = average_precision_score(test_y, probs[:, 1],
                                                             average="micro")

        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[0], precision[0], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
        plt.legend(loc="lower left")
        plt.show()

        # Plot Precision-Recall curve for each class
        plt.clf()
        plt.plot(recall["micro"], precision["micro"],
                 label='micro-average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]))
        # for i in range(n_classes):
        for i in range(2): #same deal
            plt.plot(recall[i], precision[i],
                     label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(i, average_precision[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="lower right")
        plt.show()
    
def test_on_train():
    
    global data
    classifier = LogisticRegression()
#     classifier = LinearRegression()
    
    #train on whole dataset then test on dataset
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
    
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    probs = classifier.fit(train_x, train_y).predict_proba(train_x)
    fpr, tpr, thresholds = roc_curve(train_y, probs[:, 1])
#     probs = classifier.fit(train_x, train_y).predict(train_x)
#     fpr, tpr, thresholds = roc_curve(train_y, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression Test on Train') #todo
    plt.legend(loc="lower right")
    plt.show()

def rfc_test_on_train():
    
    global data
    classifier = RandomForestClassifier()
    
    #train on whole dataset then test on dataset
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
    
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    probs = classifier.fit(train_x, train_y).predict_proba(train_x)
    fpr, tpr, thresholds = roc_curve(train_y, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RFC Test on Train')
    plt.legend(loc="lower right")
    plt.show()

def bootstrap(n_percent, m_times):

    global data
    classifier = LogisticRegression()
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i in range(m_times):
        test_x, test_y, train_x, train_y = pd.bootstrap_sampling(data, n_percent, i) #use i as seed
        probs = classifier.fit(train_x, train_y).predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        plt.plot(fpr, tpr, lw=1) #lets not do labels

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= m_times
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Bootstrap %d percent of data %d times (SCOREDATA.vina.balanced)' % (n_percent, m_times))
    plt.legend(loc="lower right")
    plt.show()
        
def leave_target_out():
    global data
    classifier = LogisticRegression()
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    
    targets = list(data.keys())
    
    #added to try and fix legend
    fig = plt.figure()
    ax = plt.subplot(111)
    
    for i in range (len(targets)):
        test_x, test_y, train_x, train_y = pd.leave_one_target_out(data, i)
        probs = classifier.fit(train_x, train_y).predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1])
        
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, label='%s (area = %0.2f)' % (targets[i], roc_auc)) 
        
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(targets) 
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Leave one (target) out (SCOREDATA.vina.balanced)')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=4)

    #plt.legend(loc="lower right")
    plt.show()

def leave_target_out_dist():
    global data
    classifier = LogisticRegression()
    
    rocs = []
    targets = list(data.keys())
    
    for i in range (len(targets)):
#     for i in range (3):
        test_x, test_y, train_x, train_y = pd.leave_one_target_out(data, i)
        probs = classifier.fit(train_x, train_y).predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1])
#         rocs.append((targets[i], auc(fpr, tpr)))
        rocs.append(auc(fpr, tpr))
    
#     sorted_rocs = sorted(rocs, key=lambda x: x[1])
#     for tuple in sorted_rocs:
#         print tuple

    plt.hist(rocs)
    plt.title("Target AUC distribution")
    plt.xlabel("AUC")
    plt.ylabel("Frequency")
    plt.show()
    
    
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="input file of scoredata")
file = parser.parse_args().filename
print file
data = pd.create_dict(file)

#TODO - take input file as commandline arg and update titles accordingly based on that
#data = pd.create_dict('SCOREDATA.vina.balanced')
#data = pd.create_dict('SCOREDATA.vina.reduced')
#data = pd.create_dict('SCOREDATA.dkoes.reduced')


#linreg_ccv_plot_roc(10)
#precision_recall_curve(10)
#rfc_test_on_train()
#bootstrap(10, 100)
#leave_target_out()
#ccv_plot_roc(10)

#logreg_precision_recall_ccv(10)

##try next...
#ccv_plot_roc(10)





