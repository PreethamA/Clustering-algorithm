#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Possibilistic Fuzzy C Mean Clustering Algorithm (PFCM)

References

[1] Nikhil R. Pal , Kuhu Pal,James M.keller and James C.Bezdek
A Possibilistic Fuzzy c-Means Clustering Algorithm

[2] Scipy and Numpy Documentation
https://docs.scipy.org/doc/

[3] Python Documentation
https://docs.python.org/3/index.html

'''


import numpy as np
from scipy.spatial.distance import cdist

import argparse


def gamma_value(u, d, m):
    
    '''
    calculation of gamma values using equation(2) from[1] for PFCM
    
    Parameters::
    u:float
    membership value of fcm
    d:float
    equlidiean distance between data and prototype
    m:float
    fuzzifer value
    
    Attributes::
    n:float
    gamma values 
    '''
    u = u ** m
    # aviod zeros 
    u = np.fmax(u, np.finfo(np.float64).eps)
    d2=d**2
    n = np.sum(u * d2, axis=1) / np.sum(u, axis=1)

    return n







def update_clusters(x, u, m):
    '''
    Parameters::
    x:float
    data points    
    
    u:float
    membership value of fcm

    m:float
    fuzzifer value 
    
    Attributes::
    v:float
    cluster centres(prototypes)  of FCM
    um:float
    membership value power to fuzzifer for FCM costfunction
    '''
    um = u ** m
    
    v = um.dot(x) / np.atleast_2d(um.sum(axis=1)).T
  
    return um, v

def update_pfcmclusters(x, u,typ, m,a,b,eta):
    '''
    Paratmeters::
    x:float
    data point values
    
    u:float
    membership value of PFCM
    typ:float
    typicality value of PFCM
    
    m:float
    fuzzifer value
    a:float
    constant for the memberships
    b:float
    constant for the typicallity
    eta:float
    user-defined constant for typucality
    
    
    Attributes::
    v:float
    prototypes(cluster centres) values for PFCM
    um:float
    membership values power to fuzzifer  for PFCM costfunction
    k:float
    membership values and typicallity constant mulitplication for PFCM costfunction
    '''
    um = u ** m
    typeta = typ ** eta
    aum=a*um

    btypeta=b*typeta
    k=aum+btypeta

    v=k.dot(x)/np.atleast_2d(k.sum(axis=1)).T
    
    return um, v,k

def membership(x, v, m, metric):
    '''
    Parameters::
    x:float
    data values   
    v:float
    prototypes(cluster centres) values
    m:float
    fuzzifer value 

    metric:float
    metric norm is equlidean distance


        
    
    Attributes::
    d:float
    euclidean distance between cluster centre and data point 
    u:float
    membership values 
    '''
    
    # calculating equidistance
    d = cdist(x, v, metric=metric).T

    # Sanitize equidistances (Avoid Zeroes)
    d = np.fmax(d, np.finfo(np.float64).eps)
    # power
    exp = -2. / (m - 1)
    d2 = d ** exp
    
    u = d2 / np.sum(d2, axis=0, keepdims=1)
    
    return u, d

def typicality(x, v, gamma, m, metric,b,eta):
    '''
    Paraments::
    x:float
    data values
    v:float
    prototypes(cluster centres) values of PFCM    
    gamma:float
    gamma values 
    m:float
    fuzzifer value
    metric:float
    metric norm is equlidean distance 
    b:float
    constant for the typicallity
    eta:float
    user-defined constant for typucality
     
    
    Attributes::
    t:float
    Typicallity values of PFCM 
    d:float
    euclidean distance between cluster centre and data point 
    '''
    d = cdist(x, v, metric=metric)
    
    d = np.fmax(d, np.finfo(np.float64).eps)
    #d[d==0]=1
    d1 = b / gamma

    power=(d ** 2)

    d2=d1*power

    exp = 1. / (eta - 1)

    d2 = d2 ** exp

    addone=(1. + d2)

    #typicallity
    t= 1. / addone

    return t.T,d
    
def fcm(data, cluster, fuzzifer, tolerance, max_iterations, metric="euclidean", v0=None):
    '''
    Parameters::
    data:float
    data values
    cluster:int
    number of prototypes(clusters)
    tolerance:float
    tolerance for the convergence of alogrithm 
    max_iteration:int
    number of iteration for the optimization problem
    metric:float
    metric norm is equlidean distance
    v0:float
    random number selection of cluster centre from the given data 
    
    Attributes::
    u:float
    final membership values  
    V:float
    final prototypes(cluster centres)  
    d:float
    euclidean distance between cluster centre and data point 
    '''
    
    
    ## check the length of data
    if not data.any() or len(data) < 1 or len(data[0]) < 1:
        print("Error: Data is in incorrect format")
        return

    # size  of  Features, Datapoints in data
    S, N = data.shape
    ## check the number of cluster 
    if not cluster or cluster <= 0:
        print("Error: Number of clusters must be at least 1")
    ## check the Fuzzifer value 
    if not fuzzifer:
        print("Error: Fuzzifier must be greater than 1")
        return

    # Initialize the cluster centers
   
    if v0 is None:
        # Pick random values from dataset

        v0 = data[np.random.choice(data.shape[0], int(cluster), replace=False), :]
        print('fcm intial centres \n',v0)
    # List of all cluster centers (Bookkeeping)
    v = np.empty((max_iterations, int(cluster), N))

    v[0] = np.array(v0)

    # Membership Matrix Each Data Point in eah cluster
    u = np.zeros((max_iterations, int(cluster), S))
    
    # Number of Iterations
    t = 0

    while t < max_iterations - 1:
        # upadting membership values
        u[t], d = membership(data, v[t], fuzzifer, metric)
        # upadting cluster centers(prototypes) values

        um,v[t + 1] = update_clusters(data, u[t], fuzzifer)
  
        costfunction=um*d

        # Stopping Criteria 
        if np.linalg.norm(v[t + 1] - v[t]) < tolerance:

            break

        t += 1
        

    return v[t], u[t - 1],d

def pfcm(data, cluster, fuzzifer, tolerance, max_iterations,a,b,eta,gamma, metric="euclidean", v0=None):
    '''
    Parameters::
    data:float
    data points    
    cluster:int
    number of clusters(prototypes)
    fuzzifer:float
    fuzzifer value
    tolerance :float
    tolerance for the convergence of alogrithm
    max_iteration:int
    number of iteration for the optimization problem
    a:float
    weight(constant) for membership in PFCM
    b:float
    weight(constant) for typicality in PFCM
    eta:float
    user defined constant for typicality
    gamma:float
    calculated using FCM cluster centers and distance between prototypes and data points
    metric:float
    metric norm is equlidean distance
    v0:float
    random number selection of cluster centre from the given data or FCM cluster center,default is random from the given data    
    
    
    Attributes::
    u[t - 1]:float
    final membership values of PFCM
    typ[t]:float
    final typicality values of PFCM
    v[t]:float
    final cluster centres values of PFCM
    
    '''
    
    ## check the length of data
    if not data.any() or len(data) < 1 or len(data[0]) < 1:
        print("Error: Data is in incorrect format")
        return

    # Num Features, Datapoints
    S, N = data.shape
    ## check the number of cluster
    if not cluster or cluster <= 0:
        print("Error: Number of clusters must be at least 1")
    ## check the fuzzifer value
    if not fuzzifer:
        print("Error: Fuzzifier must be greater than 1")
        return

    # Initialize the cluster centers random from data  if FCM cluster is not given 
    
    if v0 is None:

        v0 = data[np.random.choice(data.shape[0],int(cluster), replace=False), :]
        print(' pfcm intial centres \n',v0)
    else:
    
        print(' pfcm intial centres \n',v0)
    #  cluster centers     
    v = np.empty((max_iterations, int(cluster), N))
 
    v[0] = np.array(v0)
 
    # Membership matrix Each Data Point in eah cluster
    u = np.zeros((max_iterations, int(cluster), S))
    
   
    t = 0
    # typicallity matrix Each Data Point in eah cluster
    typ= np.zeros((max_iterations,int(cluster), S))
    while t < max_iterations - 1:
        # upadting membership values
        u[t], d = membership(data, v[t], fuzzifer, metric)
        # upadting typicality values
        typ[t],d=typicality(data, v[t], gamma, fuzzifer, metric,b,eta)
        # upadting cluster centers(prototypes) values
        um,v[t + 1],k = update_pfcmclusters(data, u[t],typ[t], fuzzifer,a,b,eta)
   
        penality=(1-typ[t])**eta
       
        costfunction=k*(d.T**2)+gamma.sum()*penality

        # Stopping Criteria
        if np.linalg.norm(v[t + 1] - v[t]) < tolerance:

            break

        t += 1
        if t==max_iterations - 1:
           typ[t]=typ[t-1] 

    return v[t],  u[t - 1],  typ[t]

def main():
    '''
    Parameters::

    data:float
    data values,data parsing through text file,rows vectors seperated by comma(,)    
    c:int
    number of clusters(prototypes)(default value 2)
    e:float
    tolerance for the convergence of algorithm(default value 1e-5)
    max_iteration:int
    number of iteration for the optimization problem(default value 100)
    m:float
    fuzzifer(default value 2)
    a:float
    constant for the membership(default value 2)
    b:float
    constant for the typicallity(default value 2)
    eta:float
    constant(default value 2)
    v0:float
    selection of cluster centre random from the given data or FCM cluster center (default random)     
    
    Attributes(displays)::
        
    displays membership,data,typicality,cluster center(prototypes),constants of FCM and PFCM algorithm 
    
    '''
    # argument parsing values
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',  help="data in the text file ,only numbers,numbers should be seprated by comma(,),no Default value")
    parser.add_argument('-a',  help="constant or weight,Default value is 0.5",type=float, default=0.5)
    parser.add_argument('-b',  help="constant or weight,Default value is 0.5",type=float, default=0.5)
    parser.add_argument('-m',  help="fuzzifer value,Default value is 2",type=float, default=2)
    parser.add_argument('-t',  help="epsilon or torleance value to the terminate the iterations,Default value is 1e-5",type=float, default=1e-5)
    parser.add_argument('-itera', help=" maximum iterations for clustering the data,Default value is 100",type=int, default=100)
    parser.add_argument('-eta',  help=" eta value constant ,Default value is 2",type=float, default=2)
    parser.add_argument('-c',  help="number of prototypes,Default value is 2",type=int, default=2)
    parser.add_argument('-initial', help="initial cluster centers two options first 'None' for random selection,second 'fcm' for fcm clustercenter,Default value is None",type=str, default=None)
    parsed = parser.parse_args()
    #iris data with labeled data to skip the labels
    #data=np.loadtxt(parsed.data, delimiter=',',usecols=(0,1,2,3))
    #print('data',parsed.data)
    
    
    # data from the text file
    data=np.loadtxt(parsed.data, delimiter=',')
    print('data \n',data)
    # number of clusters
    print('\n no of clusters:',parsed.c)
    
    # Fuzzy C Mean algorithm
    # input data,no of clusters,fuzzifer,tolerance,iteration,metric,intial cluster center
    # output  final membership,final cluster centers and fcm_equidistance 
    fcm_cluster_center_final, fcm_membership_final,fcm_equidistance =fcm(data,parsed.c, parsed.m, parsed.t, parsed.itera, metric="euclidean", v0=None)
	# Fuzzy C Mean final cluster centers
    print("fcm final centres \n",fcm_cluster_center_final)
    #print('FCM membership \n',fcm_membership_final.T)
    # weight constants (a,b) print warning if greater than one 
    
    if parsed.a+parsed.b>1:
        print('\n warning a+b>1')
        print('\n a:',parsed.a)
        print('\n b:',parsed.b)
        print('\n a+b:',parsed.a+parsed.b)
        
    else:
        print('\n a:',parsed.a)
        print('\n b:',parsed.b)
    
    #print('\n fuzzifer:',parsed.m)
    #print('\n eta:',parsed.eta)    
    # gamma values are calculated using Fuzzy C Mean cluster centers 
    #membership u,distance d,fuzzifer m default value 2
    gamma1 = gamma_value(fcm_membership_final, fcm_equidistance, parsed.m)
    print('\n Gamma:',gamma1)
     # PFCM intial cluster can be random initialization from data or Fuzzy C Mean cluster centers(prototypes)
    if parsed.initial=='fcm' :
       pfcm_cluster_center_initial=fcm_cluster_center_final

    else:
       pfcm_cluster_center_initial=None 

    # Possibilistic Fuzzy C Mean (PFCM) Algorithm function call
    # input data,no of clusters,fuzzifer,tolerance,iteration,metric,intial cluster,weight constants(a,b),gamma,eta
    # output intial and final (membership and cluster centers),typicallity
    pfcm_cluster_center_final,  pfcm_membership_final,pfcm_typicality_final=pfcm(data,parsed.c, parsed.m, parsed.t,parsed.itera,a=parsed.a,b=parsed.b,eta=parsed.eta,gamma=gamma1, metric="euclidean", v0=pfcm_cluster_center_initial)
    # prints data,final cluster,membership,typicality 
    
    print('PFCM final cluster centre \n',pfcm_cluster_center_final)
    print('PFCM membership \n',pfcm_membership_final.T)
    print('PFCM typicality \n',pfcm_typicality_final.T)
    


    return 0

if __name__ == '__main__':
    main()
    
