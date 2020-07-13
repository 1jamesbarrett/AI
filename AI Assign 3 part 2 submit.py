# AI Assign 3, part 2
# Remember to add the vector 1 (intercept) ahead of your data matrix. 
# They represent age (years), and weight (kilograms and last column is label, the height (meters)
# seems we should scale variables, likley use mean and standard deviation of each feature
# Scale each feature (i.e. age and weight) by its (population) standard deviation, and set its mean to zero.
# scale = x-mu / stand dev
# . Initialize your β’s to zero
# run the algorithm for exactly 100 iterations with each alpha
# Compare the convergence rate when α is small versus large. What is the ideal learning rate to obtain an accurate model?
# In addition to the nine learning rates above, come up with your own choice of value for the learning rate.
# numpy, scipy, pandas and scikit-learn
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

alphas  = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 11] # note hy curley bracket on orig
# remember to add one of my own rates
################################################################################################
# read into df
#inputs = pd.read_csv('C:/barrett/Edx/AI/Assign 3/input2.csv', header=None,  names = ['age_years','weight_kilo', 'height_meters']) ##, skiprows=0)
inputs = pd.read_csv('input2.csv', header=None,  names = ['age_years','weight_kilo', 'height_meters']) ##, skiprows=0)
###################################################################################################
# scale data and add column for intercept
input_file_scaled =  preprocessing.scale(inputs)


#insert value of oneinto new column at postion 0 for the intercept
#print(input_file_scaled)

input_file_scaled = np.insert(input_file_scaled ,0,1, axis=1)
#print(input_file_scaled)
###################################################################################################
# linear regression via gradient descent
##################################################################################################
#output_file = 'C:/barrett/Edx/AI/Assign 3/output2.csv'
output_file = 'output2.csv'
file_out = open(output_file, 'w')
i= 0
jb_iter  = [100, 100, 100, 100, 100, 100, 100, 100, 100, 200]

while i < 10:
    alpha_lp = alphas[i]
    jb_iter_lp = jb_iter [i]
    X = input_file_scaled[:,0:3]
    Y = input_file_scaled[:,3:4]
    linreg_lp = linear_model.SGDRegressor(alpha = alpha_lp, fit_intercept=False,  n_iter=jb_iter_lp, penalty = None, warm_start=False, verbose=0)
    linreg_lp.fit(X,Y.ravel())
    #print(linreg_lp.coef_[0], linreg_lp.coef_[1], linreg_lp.coef_[2])

    file_out.write("%f, %d, %1.3f, %1.3f, %1.3f\n"%(alpha_lp, jb_iter_lp, linreg_lp.coef_[0], linreg_lp.coef_[1],linreg_lp.coef_[2]))
    i=i+1
# propper order alpha, number_of_iterations, b_0, b_age, and b_weight 
####################################################################################################
# http://scikit-learn.org/stable/modules/preprocessing.html
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor

##########################################
# code used during dev
#  check slices
#input_file_scaled
#print (input_file_scaled[:,0:3])
#print('break 1')
#print (input_file_scaled[:,3:4])
#print('break 2')
#print (input_file_scaled)

# assign is much easier in dataframe
#file_size = len(input_file)
#print (file_size)
#inter_cept = np.ones(file_size)
#print(inter_cept)
#print(input_file)
