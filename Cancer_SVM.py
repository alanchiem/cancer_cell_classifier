#  SVM (Support Vector Machines) to build and train a model using human cell records, 
#  and classify cells to whether the samples are benign or malignant.

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import requests
# dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv"
response = requests.get(url)

from io import StringIO
data = StringIO(response.text)
cell_df = pd.read_csv(data)

# *** Pre-processing and selection
print(cell_df.head())

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)


feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print('X: ', X[0:5])

# We want the model to predict the value of Class (that is, benign (=2) or malignant (=4)). 
# As this field can have one of only two possible values, we need to change its measurement level to reflect this.
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
print('y: ', y[0:5])

# *** Train / Test Data

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# *** Modeling (SVM with SciKit Learn)

# Kernelling: mapping data into a higher dimensional space 'linear', 'poly', 'rbf', 'sigmoid'
# THE GOAL: Test different kernel functions by comparing their confusion matrix, f1-score, jaccard-index

f1_scores = []
jaccard_scores = []

# Prepare Evaluation Tools
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(kernel, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print(kernel + ' confusion matrix, without normalization')

    print(cm)


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# *** Start the Modeling / Evaluation

from sklearn import svm
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for i, k in enumerate(kernels):
    clf = svm.SVC(kernel = k)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)

    # Evaluation

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(k, cnf_matrix, classes=['Benign','Malignant'],normalize= False,  title='Confusion matrix - ' + k)
    plt.show()
    
    # f1-score
    from sklearn.metrics import f1_score
    f1_scores.append(f1_score(y_test, yhat, average='weighted'))
    # print('f1-score: ', f1_score(y_test, yhat, average='weighted'))

    # jaccard-index
    from sklearn.metrics import jaccard_score
    jaccard_scores.append(jaccard_score(y_test, yhat,pos_label=2))
    # print('jaccard_score: ', jaccard_score(y_test, yhat,pos_label=2))

print('f1 scores: ', f1_scores)
print('jaccard scores: ', jaccard_scores)

plt.bar(kernels, f1_scores, color='maroon', width=0.4)
plt.title('f1 scores')
plt.xlabel('kernel function')
for i in range(len(kernels)):
        plt.text(i, f1_scores[i], str(f1_scores[i])[:5], ha = 'center')
plt.show()
    
plt.bar(kernels, jaccard_scores, color='maroon', width=0.4)
plt.title('jaccard index scores')
plt.xlabel('kernel function')
for i in range(len(kernels)):
        plt.text(i, jaccard_scores[i], str(jaccard_scores[i])[:5], ha = 'center')
plt.show()


