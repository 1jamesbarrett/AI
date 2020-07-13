## ML Assign 3
# K-means and EM Gaussian mixture models
#
# python hw3_clustering.py X.csv
#
# You should implement both the K-means and the EM-GMM algorithms either in the "hw3_clustering" file
#   or have your "hw3_clustering" call your implementations of these algorithms located in different files.
# X.csv: A comma separated file containing the data. Each row corresponds to a single vector
#
# learn 5 clusters. Run both algorithms for 10 iterations
#
# You can initialize your algorithms arbitrarily
#     We recommend that you initialize the K-means centroids by randomly selecting 5 data points
#     For the EM-GMM, we also recommend you initialize the mean vectors in the same way,
#     and initialize (pi-like) to be the uniform distribution
#     and each sigma to be the identity matrix.
# Ouput
#   Where you see [iteration] and [cluster] below, replace these with the iteration number and the cluster number
#  http://archive.ics.uci.edu/ml/machine-learning-databases/00396/ copy name x.csv
###############################################################
# output
# centroids-[iteration].csv:
# pi-[iteration].csv
# mu-[iteration].csv
# Sigma-[cluster]-[iteration].csv:
###############################
# imports
from __future__ import division
import sys

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import multivariate_normal as MV_N

# file read into pandas
X=(pd.read_csv(sys.argv[1],header=None))
#X=  pd.read_csv ('C:/barrett/Edx/Machine Learning/Assign 3/X.csv', header=None, usecols=[1,2,3,4,5,6,]) #7,8,9,10,11,12,13])#,11,12,13,14,15,16,17,18,19,20])
X=X.fillna(X.mean())
#print(X)
#############################################################################################################
## K-Means Clustering
def Find_Best_Cluster(X,k,init):
    N=len(X)
    Item_Class={}
    for row in range(0,N):
        List=[]
        for cls in range(0,k):
            Dist=distance.euclidean(X.loc[row],init.ix[cls])
            List.append(Dist)
        p=np.argmin(List)
        Item_Class[row]=p
    return Item_Class

def Find_Centroids(X,k,number_iters):
    N=len(X)
    init=X.sample(k).reset_index(drop=True)
    for iter in range(0,number_iters):
        Cluster=Find_Best_Cluster(X,k,init)
        Data=pd.DataFrame.from_dict(Cluster,orient='index')
        X['class']=Data[0]
        init=X.groupby('class').mean()
        del X['class']
        File_Out_Cen='centroids-'+ str(int(iter)+1)+'.csv'
        init.to_csv(File_Out_Cen, header=False,index=False)
##################################################################
Find_Centroids(X,5,10)
##################################################################

##################################################################
def Initialize(X,k):
    d=X.shape[1]
    mu_0=X.sample(k).reset_index(drop=True)
    #mu_0=pd.DataFrame(np.random.rand(k,d))
    sigma0=X.cov()
    prior=[1/k]*k
    columns=np.arange(k)
    df =pd.DataFrame(index=X.index,columns=columns)
    df= df.fillna(1/k)
    for row in range(0,len(X)):
        L=[]
        for i in range(0,k):
            g=MV_N.pdf(X.loc[row],mu_0.loc[i],sigma0)*prior[i]
            L.append(g)
        #W = [float(i)/sum(L) for i in L]
        W = [float(i)/(sum(L) +.00001)for i in L]
        df.loc[row]=W
    return df,mu_0
##################################################################

def Maximum(X,k,df):
    prior=df.mean()
    e=np.exp(-10) # -100
    #mu=Initialize(X,k)[1]
    #mu = pd.DataFrame(np.empty((k,X.shape[1],)))
    s=(k,X.shape[1])
    #print('shape'+str(s))
    mu=pd.DataFrame(np.ones(s))
    #print(mu)
    for i in range(0,k):
        #mu.loc[i]=np.dot(df[i],X)/(sum(df[i])+e)
        #L=dot_product(df[i],X)
        df=df.fillna(0)
        X=X.fillna(X.mean())
        L=np.matrix(df[i].T)*np.matrix(X)
        #print('L:'+str(L))
        mu.loc[i]=L/sum(df[i]+e)
        mu=mu.fillna(1)

    D={}
    for cls in range(0,k):
        M=0
        for i in range(0,len(X)):  # goes through every line of data aprox 810 in sample
            '''
            print(df.loc[i][cls])
            print(' piece 1')
            print(np.matrix(X.loc[i]))
            print(' piece 2 A')
            print((mu.loc[cls]).T)
            print(' piece 2')
            print(np.matrix(X.loc[i]-mu.loc[cls]))
            '''
            #print(' piece 3')
            x_spot = X.loc[i]
            mu_spot = mu.loc[cls]
            x_mu_spot = np.subtract(x_spot, mu_spot)
            #M=M+df.loc[i][cls]*(np.matrix(X.loc[i]-mu.loc[cls]).T * np.matrix(X.loc[i]-mu.loc[cls]))
            #print('x_mu_spot')
            #print(x_mu_spot)
            #print('x_mu_spot')
            #print(np.matrix (x_mu_spot).T)
            M=M+df.loc[i][cls]* (np.matrix (x_mu_spot).T * np.matrix (x_mu_spot))
            #df times (x-mu)transpose times (x-mu)
        D[cls]=M
    #print('D Len')
    #print(len(D[0]))
    #print('D Start')
    #print(D)

    #print('D End')
    return prior,mu,D
#####################################################################

# learn 5 clusters. Run both algorithms for 10 iterations
k=5
number_iters=10
df=Initialize(X,k)[0]
#print(df)


for iter in range(0,number_iters):
    #print('iter:'+str(iter))
    Max_One = Maximum(X,k,df)
    #print(' Max_One Start ')
    #print(Max_One)
    #print(' Max_One End')
    prior=Max_One[0]
    prior_df=pd.DataFrame(prior)
    File_Out_Pi='pi-'+ str(int(iter)+1)+'.csv'
    prior_df.to_csv(File_Out_Pi,header=False,index=False)
    mu=Max_One[1]
    #print('mu:'+str(mu))
    File_Out_Mu='mu-'+str(int(iter)+1)+'.csv'
    mu.to_csv(File_Out_Mu,header=False,index=False)
    #print('mu:')
    #print(mu)
    #print('find this spot becuase sigma ia defined here')
    sigma=Max_One[2]
    #print('sigma value:')
    #print(sigma)

    for cluster in range(0,len(sigma)):
        #print('cluster:'+str(cluster))
        data=pd.DataFrame(sigma[cluster])
        #print('cluster val' + str(int(cluster)))
        #print( str(int(cluster) +1))
        File_Out_Sigma='Sigma-'+ str(int(cluster) +1)+'-'+ str(int(iter) +1)+'.csv'

        data.to_csv(File_Out_Sigma,header=False,index=False)
        for row in range(0,len(X)):
            #print(len(X))
            #print('len x')
            L=[]
            for i in range(0,k):
                '''
                print('start check last section')
                print ('Cluster:'+str(i))
                print(row)
                print('row')
                
                print (X.loc[row])
                print('X.loc[row]')
                print(mu.loc[i])
                print('mu.loc[i]')
                print('sigma[i]) start')
                print(sigma[i])
                print('sigma[i]) end')
                '''

                g=0.001
                L.append(g)
            W = [float(i)/(sum(L) +.00001) for i in L]
            #print('phi:-'+str(W))
        df.loc[row]=W


############################################
#############################################
# ref
# with multivariate_normal from SciPy (from scipy.stats import multivariate_normal)
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html
# https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory






