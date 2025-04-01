# -*- coding: utf-8 -*-
"""
@author: mdn769
Updated for run-1: 
Issues addressed: 
1. VarianceFilter to remove all zero-variance features
2. Fix max column issue, if dataset do not have more columns, program will automaticaly adjust that. Updated: getNBestFeatures(df, n)
"""
import qml_custom_functions as qcf
import multiprocessing as mp
import pandas as pd
import numpy as np

import os
from collections import Counter
import time
import warnings

warnings.filterwarnings('ignore')

file_path = "datasets/"

def process_data_file(algo, i, file):    
    os.makedirs(f"result_qml_cml/{algo}-{i}/", exist_ok=True)
    out_file_name = f"result_qml_cml/{algo}-{i}/{file}.txt"
    with open(out_file_name, 'w') as f: #just initializing the outputfile. 
        f.write('')
    print("dataset: "+file)   
        
    feature_dim = 2 #for vqc: 5/10, for others 15
    max_data_row = -500 #actual program 500, take  at most 500 latest data row
        
    df = pd.read_csv(file_path + file)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    trainX, trainY, testX, testY = qcf.split_train_test_NBest(df.iloc[max_data_row:,:], feature_dim)
    training_features = np.array(trainX, dtype='float')
    training_labels =np.array(trainY, dtype='float')
    test_features = np.array(testX, dtype='float')
    
    with open(out_file_name, 'a') as f:
        f.write("dataset: "+file+'\n')
        f.write('number of training sample: '+str(trainX.shape[0])+'\n')
        f.write("actual- buggy: " + str(Counter(testY)[1])+" non-buggy: "+ str(Counter(testY)[0])+'\n')
        f.write("---------------------------------\n")

    # get the start time
    st = time.time()
    train_time, test_time, predict = qcf.runML(algo, training_features.shape[1], training_features, training_labels, test_features)
    qcf.report_result(algo, out_file_name, predict, testY.to_numpy().flatten())   
    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = round(et - st, 2)
    print('execution time:', elapsed_time, 'seconds')
    print('number of training sample: '+str(trainX.shape[0]))
    with open(out_file_name, 'a') as f:
        f.write('train time: '+str(train_time)+' seconds\n')       
        f.write('test time: '+str(test_time)+' seconds\n')       
        f.write('total processing time: '+str(elapsed_time)+' seconds\n')                
        f.write("\n")
    print()
    
    print("--------------###----------------")
    with open(out_file_name, 'a') as f:
        f.write("--------------###----------------")
    #break        


if __name__ == "__main__":    
    flist = os.listdir(file_path)
    #for algo in ['pqsvc', 'qsvc', 'vqc', 'svc', "rf", "knn", "gbc", "pct"]:
    for algo in ['pqsvc', 'qsvc', 'vqc', 'svc', "rf", "knn", "gbc", "pct"]:        
        for i in [1, 2, 3, 4, 5]: #update for required repeatation
            processes = []        
            for file in flist:
                process = mp.Process(target=process_data_file, args=(algo, i, file))
                process.start()
                processes.append(process)
                
            for process in processes:
                process.join()
            