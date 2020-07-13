#  if I have to chose the one that have the best mean_train_score and there are many model that have equal values what should I chose?
#       test_score matters
"""
retrieving the training accuracy of the best model?
Look at the value of the model object that you applied the fit method too. In this case my object was for logistic:
maxv=max(clflog.scores_.values())
or for knn
clfknn.best_score
"""
# AI Assign 3, part 2
#
# Make a scatter plot of the dataset showing the two classes with two different patterns.
#
# do not need to scale/normalize the data for this question
# split your data into training (60%) and testing (40%)
# Train-test splitting and cross validation functionalities are all readily available in sklearn.
#
#
# use the support vector classifiers in the sklearn package
#    Use SVM with different kernels to build a classifier
#    use stratified sampling (i.e. same ratio of positive to negative in both the training and testing datasets).
#    Use cross validation (with the number of folds k = 5) instead of a validation set.
#
# SVM with Linear Kernel. Observe the performance of the SVM with linear kernel.
# Search for a good setting of parameters to obtain high #classification accuracy.
# Specifically, try values of C = [0.1, 0.5, 1, 5, 10, 50, 100].
#     Read about sklearn.grid_search and how this #can help you accomplish this task.
#    After locating the optimal parameter value by using #the training data, record the #corresponding best score (training data
#    accuracy) achieved. Then apply the testing data to the model, #and record the actual test #score. Both scores will be a number
#    between zero and one.
#
#SVM with Polynomial Kernel. (Similar to above).
#Try values of C = [0.1, 1, 3], degree = [4, 5, 6], and gamma = [0.1, 0.5].
#
#SVM with RBF Kernel. (Similar to above).
#Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100] and gamma = [0.1, 0.5, 1, 3, 6, 10].
#
#Logistic Regression. (Similar to above).
#Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100].
#
#k-Nearest Neighbors. (Similar to above).
#Try values of n_neighbors = [1, 2, 3, ..., 50] and leaf_size = [5, 10, 15, ..., 60].#
#
#Decision Trees. (Similar to above).
#Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].
#
#Random Forest. (Similar to above).
#Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].

##########################################################################################

##########################################################################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def main():
    # A	B	label
    # input_file = pd.read_csv('C:/barrett/Edx/AI/Assign 3/input3.csv', header=0 ) # header 0 for yes, 1 for no
    #input_file = np.genfromtxt('C:/barrett/Edx/AI/Assign 3/input3.csv', skip_header=1,  delimiter=',') #names=True,
    input_file = np.genfromtxt('input3.csv', skip_header=1,  delimiter=',') #names=True,
    #print(input_file[:,0]) # grabs first column

    #output_file = 'C:/barrett/Edx/AI/Assign 3/output3.csv'
    output_file = 'output3.csv'
    file_out= open(output_file, 'w')

    x = input_file[:, :-1]
    y = input_file[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=67, stratify=y)
    # train_size is another possible input
    #stratified sampling
    ###################################################################################
    # 'kernel': ['linear'
    # try C = [0.1, 0.5, 1, 5, 10, 50, 100
    params = {'C': [0.1, 0.5, 1, 5, 10, 50, 100],  'kernel': ['linear']}
    grid_srch = GridSearchCV(SVC(), params, n_jobs=1)
    grid_srch.fit(x_train, y_train)
    file_out.write("%s, %0.2f, %0.2f\n"%('svm_linear', grid_srch.best_score_, grid_srch.score(x_test, y_test)))

    ###################################################################################
    # Kernel = Poly
    #Try values of C = [0.1, 1, 3], degree = [4, 5, 6], and gamma = [0.1, 0.5].
    params = { 'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 0.5], 'kernel': ['poly']}
    # degree for poly only
    grid_srch = GridSearchCV(SVC(), params, n_jobs=1)
    grid_srch.fit(x_train, y_train)
    file_out.write("%s, %0.2f, %0.2f\n"%('svm_polynomial', grid_srch.best_score_, grid_srch.score(x_test, y_test)))

    ###################################################################################
    # Kernel = rbf
    #Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100] and gamma = [0.1, 0.5, 1, 3, 6, 10].
    params = {    'C': [0.1, 0.5, 1, 5, 10, 50, 100],   'gamma': [0.1, 0.5, 1, 3, 6, 10],'kernel': ['rbf'] }
    grid_srch = GridSearchCV(SVC(), params, n_jobs=1)
    grid_srch.fit(x_train, y_train)
    file_out.write("%s, %0.2f, %0.2f\n"%('svm_rbf', grid_srch.best_score_, grid_srch.score(x_test, y_test)))

    ###################################################################################
    # Solver = liblinear Logistic
    #Try values of C = [0.1, 0.5, 1, 5, 10, 50, 100].
    params = { 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'solver': ['liblinear'] }
    grid_srch = GridSearchCV(LogisticRegression(), params, n_jobs=1)
    grid_srch.fit(x_train, y_train)
    file_out.write("%s, %0.2f, %0.2f\n"%('logistic', grid_srch.best_score_, grid_srch.score(x_test, y_test)))

    ###################################################################################
    # n_neighbors Classifier
    # Try values of n_neighbors = [1, 2, 3, ..., 50] and leaf_size = [5, 10, 15, ..., 60].
    params = {'n_neighbors': range(1, 51), 'leaf_size': range(5, 65, 5), 'algorithm': ['auto'] }
    grid_srch = GridSearchCV(KNeighborsClassifier(), params, n_jobs=1)
    grid_srch.fit(x_train, y_train)
    file_out.write("%s, %0.2f, %0.2f\n"%('knn', grid_srch.best_score_, grid_srch.score(x_test, y_test)))

    ###################################################################################
    # Decision Tree Classifier
    # Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].
    params = { 'max_depth': range(1, 51), 'min_samples_split': range(2, 11)}
    grid_srch = GridSearchCV(DecisionTreeClassifier(), params, n_jobs=1)
    grid_srch.fit(x_train, y_train)
    file_out.write("%s, %0.2f, %0.2f\n"%('decision_tree', grid_srch.best_score_, grid_srch.score(x_test, y_test)))

    ###################################################################################
    # Random Forest Classifier
    # Try values of max_depth = [1, 2, 3, ..., 50] and min_samples_split = [2, 3, 4, ..., 10].
    params = {'max_depth': range(1, 51), 'min_samples_split': range(2, 11)}
    grid_srch = GridSearchCV(RandomForestClassifier(), params, n_jobs=1)
    grid_srch.fit(x_train, y_train)
    file_out.write("%s, %0.2f, %0.2f\n"%('random_forest', grid_srch.best_score_, grid_srch.score(x_test, y_test)))


if __name__ == "__main__":
    main()

###############################################################################################
# Ref
# https://docs.scipy.org/doc/numpy-1.10.4/reference/generated/numpy.genfromtxt.html
# http://scikit-learn.org/stable/modules/svm.html
# http://scikit-learn.org/stable/modules/svm.html#scores-probabilities
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://en.wikipedia.org/wiki/Support_vector_machine
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
