# -*- coding: utf-8 -*-
"""
Replication package for EASE 2024
"""
import qml_custom_functions as qcf
from multiprocessing import Process
import os
import pandas as pd
import numpy as np
from collections import Counter
import time
import warnings

warnings.filterwarnings('ignore')

file_path = "datasets/"
files = os.listdir(file_path)

def process_ml_result(cur_output, file, data_instance, algo, training_features_shape, training_features, training_labels, test_features, out_file_name, testY_flatten):
    # get the start time
    st = time.time()
    predict = qcf.runML(file, data_instance, algo, training_features_shape, training_features, training_labels, test_features)   
    cur_output = qcf.report_result_chunk(cur_output, data_instance, algo, out_file_name, predict, testY_flatten)   
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = round(et - st, 2)
    print('execution time:', elapsed_time, 'seconds')
    print('number of training sample: '+str(len(training_features)))
    
    cur_output += 'execution time: '+str(elapsed_time)+' seconds\n\n'              
    
    print()

    print("--------------###----------------")
    
    with open(out_file_name, 'a') as f:
        f.write(cur_output+"\n--------------###----------------")
#break

def process_file(file):
    try:
        # ... process file       
        print(file)
        out_file_name = "output/"+file+".txt"   
         #just initializing the outputfile. 
        with open(out_file_name, 'w') as f:
            f.write('')         
        print("dataset: "+file)        

        feature_dim = 15 #number of columns of the dataset
        print("Using Columns:", feature_dim)

        df = pd.read_csv(file_path + file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
            
        data_instance = 1000
        procs = []       
        while data_instance < len(df):    
            cur_output = ""                
            
            starting_point = 0
            trainX, trainY, testX, testY = qcf.split_train_test_NBest(df.iloc[starting_point:starting_point+data_instance,:], feature_dim)
            
            print("data_instance:", data_instance, "Training Instance:", trainX.shape[0])
            
            training_features = np.array(trainX, dtype='float')
            training_labels =np.array(trainY, dtype='float')
            test_features = np.array(testX, dtype='float')

            cur_output += "\ndataset: "+file+" number of instance: " +str(data_instance)+"\n"               
            cur_output += 'number of training sample: '+str(trainX.shape[0])+'\n'
            cur_output += "actual- buggy: " + str(Counter(testY)[1])+" non-buggy: "+ str(Counter(testY)[0])+'\n'
            cur_output += "---------------------------------\n"

            #run 'qsvc' separately as it takes too much time!!!
            for algo in ['pqsvc', 'vqc', 'svc', 'rf', 'knn', 'gbc', 'pct']: 
            #for algo in ['qsvc']:
                process_ml_result(cur_output, file, data_instance, algo, training_features.shape[1], training_features, training_labels, test_features, out_file_name, testY.to_numpy().flatten())
                
                
            data_instance +=200 #if you want to run multiple of times with incremental instance size!!!
                
            #for only one iteration 
            break
        # ...
    except Exception as e:
        print(str(e))      
    
    
if __name__ == "__main__":
    for file in files:
        process_file(file)

