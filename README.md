# Classifying Human Cells to Detect Cancer

Given human cell records, I used SVMs (Support Vector Machines) to train a model to classify cells on whether they are benign or malignant.

The dataset used is publicly available from the UCI Machine Learning Repository and can be seen [here](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv). It consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics.

Language/Tools: Python, SciKit Learn, Matplotlib, Pandas, Numpy

## Purpose

Kernelling is the mapping of data into a higher dimensional space to make it linearly separable. SciKit Learn has a few different kernel functions to choose from such as linear, poly, rbf, or sigmoid. 

My goal is to test the accuracy of each of these functions by comparing their confusion matrix, f1-score, and jaccard-index.

## Results

I split the dataset into train and test sets (80/20). The SVMs were trained with 546 samples and tests were run with 137 samples. 

### Confusion Matrix

The first row of the matrix is for cells that are actually benign.  
We can see that the SVM with a linear kernel function is able to correctly predict 85 as benign but mislabeled 5 as malignant. 

The second row is for cells that are actually malignant.  
The SVMs with a linear, polynomial, and radial basis function were able to fully identify every malignant cell.


Linear             |  Polynomial
:-------------------------:|:-------------------------:
<img width="636" alt="cm_linear" src="https://github.com/alanchiem/coding_practice/assets/62784950/65a3d4ec-2a3b-4ad6-9040-a56579ba3e3f"> | <img width="626" alt="cm_poly" src="https://github.com/alanchiem/coding_practice/assets/62784950/74a3342c-1ab4-40f9-abbf-62f4abfa2ca0">


Radial Basis Function          |  Sigmoid
:-------------------------:|:-------------------------:
<img width="698" alt="cm_rbf" src="https://github.com/alanchiem/coding_practice/assets/62784950/b9ef1725-8d02-474f-ac55-9cf69763c41d"> |  <img width="696" alt="cm_sigmoid" src="https://github.com/alanchiem/coding_practice/assets/62784950/61a88cc3-f5b1-490e-ba8c-ae71c836edcd">

### Other Evaluation Metrics

**F1-Score**  

The F1 score is an evaluation metric that maximizes two competing objectives, the precision and recall scores, simultaneously. It is especially useful for data sets where samples belonging to one class significantly outnumbers those found in the other class, such as the case with this dataset.  

It's calculated using the True Positives, False Positives, and False Negatives of a test as seen below.  


Precision measures how many of the “positive” predictions made by the model were correct.  
Precision = TP / (TP + FP)

Recall measures how many of the positive class samples present in the dataset were correctly identified by the model.  
Recall = TP / (TP + FN), the true positive rate  

The harmonic mean encourages similar values for precision and recall.  
f1-score = 2 x (prc x rec) / (prc + rec)

More on f1-score [here](https://www.v7labs.com/blog/f1-score-guide#:~:text=F1%20score%20is%20a%20machine%20learning%20evaluation%20metric%20that%20measures,prediction%20across%20the%20entire%20dataset.).

**Jaccard Index**

The Jaccard Index, or Jaccard Similarity, is simply the size of the intersection divided by the size of the union of the two sets: the true values and the predicted values by the classifier.  




F1-Score          |  Jaccard Index
:-------------------------:|:-------------------------:
<img width="583" alt="f1_scores" src="https://github.com/alanchiem/coding_practice/assets/62784950/3dec4088-b6f3-4887-bad2-94ec944c768a"> |  <img width="583" alt="jaccard_scores" src="https://github.com/alanchiem/coding_practice/assets/62784950/c8caa804-c9c3-43e1-9b16-acc8908a2621">


# sign_language
# cancer_cell_classifier
