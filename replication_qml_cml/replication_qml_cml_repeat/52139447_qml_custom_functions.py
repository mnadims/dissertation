import numpy as np
import pandas as pd
import time
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import Counter
from sklearn.feature_selection import VarianceThreshold
from qiskit.circuit.library import TwoLocal, ZFeatureMap, ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from qiskit_machine_learning.algorithms import PegasosQSVC, QSVC, VQC
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit import QuantumCircuit

def remove_constant_columns(X_train):
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_train)
    surviving_columns = list(X_train.columns[constant_filter.get_support()])
    X_train = constant_filter.transform(X_train)    
    return pd.DataFrame(X_train, columns=surviving_columns)
    
def train_model(X, y, feature_dim):
    seed = 1376
    algorithm_globals.random_seed = seed        
    training_features = np.array(X, dtype='float')
    training_labels = mk2DLabel(np.array(y, dtype='float'))

    feature_map=ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="linear")
    vqc = VQC(
        feature_map=ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement="linear"),
        ansatz=TwoLocal(feature_map.num_qubits, ["ry", "rz"], "cz", reps=3),
        optimizer=COBYLA(maxiter=100),
    )
    vqc.fit(training_features, training_labels)
        
    return vqc
    
def getNBestFeatures(df, n):
    y = df['contains_bug'].astype(int)
    X = df.drop(columns=['commit_hash', 'contains_bug'])
    X = remove_constant_columns(X)
    #print(X.shape[1])
    if X.shape[1] > n: #columns > n
        selector = SelectKBest(f_classif, k=n)
        selector.fit_transform(X, y)
        mask = selector.get_support()
        selected_feattures = X.columns[mask]
        return list(selected_feattures)
    else: 
        return list(X.columns)

def split_train_test_NBest(df, n):
    n_best_f = getNBestFeatures(df, n)
    ds = df[["contains_bug"]+n_best_f]

    buggy_ds = ds[ds["contains_bug"] == 1]
    clean_ds = ds[ds["contains_bug"] == 0]

    clean_downsample = resample(clean_ds,
                 replace=True,
                 n_samples=len(buggy_ds),
                 random_state=42)

    training_size = int(buggy_ds.shape[0] * 0.70)

    train_buggyX = buggy_ds.iloc[0: training_size, 1:]
    train_cleanX = clean_downsample.iloc[0: training_size, 1:]
    trainX = pd.concat([train_buggyX, train_cleanX])

    train_buggyY = buggy_ds.iloc[:training_size, 0]
    train_cleanY = clean_downsample.iloc[:training_size, 0]
    trainY = pd.concat([train_buggyY, train_cleanY])

    test_buggyX = buggy_ds.iloc[training_size: , 1:]
    test_cleanX = clean_downsample.iloc[training_size: , 1:]
    testX = pd.concat([test_buggyX, test_cleanX])

    test_buggyY = buggy_ds.iloc[training_size:, 0]
    test_cleanY = clean_downsample.iloc[training_size:, 0]
    testY = pd.concat([test_buggyY, test_cleanY])

    return trainX, trainY, testX, testY

def parity(x):
    return "{:b}".format(x).count("1") % 2

def mk2DLabel(labels1D):
    label2D = []
    for i in range(len(labels1D)):
        if labels1D[i] == 0:
            label2D.append([1, 0])
        else:
            label2D.append([0, 1])
    
    return np.array(label2D, dtype='float')

def mk1DLabel(labels2D):
    predict1D = []
    for p in labels2D:
        if p[0] == 1:
            predict1D.append(0)
        else: 
            predict1D.append(1)
      
    return predict1D

def find_prediction_stats(actual1D, predict1D, out_file_name):
    actual_class = Counter(actual1D)
    correctP = 0
    correctBuggy = 0
    correctnonBuggy = 0
    
    for i in range(len(actual1D)):
        if(actual1D[i] == predict1D[i]):
            correctP += 1
        if(predict1D[i] == 1):
            correctBuggy += 1
        elif(predict1D[i] == 0):
            correctnonBuggy +=1
    
    print("correct- total: {} ({:0.2f}%), buggy: {} ({:0.2f}%) non-buggy: {} ({:0.2f}%)\n".format(
            correctP,
            (correctP/len(actual1D))*100, 
            correctBuggy,
            (correctBuggy/actual_class[1])*100, 
            correctnonBuggy, 
            (correctnonBuggy/actual_class[0])*100))
    with open(out_file_name, 'a') as f:
        f.write("correct- total: {} ({:0.2f}%), buggy: {} ({:0.2f}%) non-buggy: {} ({:0.2f}%)\n".format(
            correctP,
            (correctP/len(actual1D))*100, 
            correctBuggy,
            (correctBuggy/actual_class[1])*100, 
            correctnonBuggy, 
            (correctnonBuggy/actual_class[0])*100))
        
def write_file_svm_light(data, label, filename):    
    with open(filename, "w") as f:
        for i in range(len(data)):
            line=""
            if label[i] == 1:
                line += '+1 '
            else:
                line += '-1 '
            for j in range(len(data[i])):
                line += str(j+1)+':'+str(data[i][j])+' '

            f.write(f"{line}\n")

def runML(algo, feature_dim, training_features, training_labels, test_features, test_labels=[], seed=1376):
    st = time.time()
    
    if algo == "svc":
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    elif algo == "pqsvc" or algo == "qsvc":        
        seed = seed
        algorithm_globals.random_seed = seed   
        # number of qubits is equal to the number of features
        num_qubits = 5
        # number of steps performed during the training procedure
        tau = 100
        # regularization parameter
        C = 1000
        feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
        qkernel = FidelityQuantumKernel(feature_map=feature_map)  
        if algo == "pqsvc":
            clf = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)   
        else: 
            clf = QSVC(quantum_kernel=qkernel)    
    #------------------------------------------  
    elif algo == 'vqc':
        num_qubits = feature_dim 
        clf = VQC(
        feature_map=ZFeatureMap(feature_dimension=num_qubits, reps=1),
        ansatz=TwoLocal(num_qubits, ["ry", "rz"], "cz", reps=3),
        optimizer=COBYLA(maxiter=100),
    )

    elif(algo=="rf"):
        clf = RandomForestClassifier(random_state=42)
        
    elif(algo=="knn"):
        clf = KNeighborsClassifier(n_neighbors=5)
        
    elif(algo=="gbc"):
        clf = GradientBoostingClassifier()
        
    elif(algo=="pct"):
        clf = Perceptron()
    
    clf.fit(training_features, training_labels)    
    train_time = round(time.time()-st, 2)
    st = time.time()
    predictions = clf.predict(test_features)
    test_time = round(time.time()-st, 2)
    return train_time, test_time, predictions

def report_result(model_name, out_file_name, predict1D, actual1D):
    result = precision_recall_fscore_support(actual1D, predict1D, average='binary')   
    print(confusion_matrix(actual1D, predict1D))
    print(result)

    with open(out_file_name, 'a') as f:
        f.write("\nclassification model: " + model_name +'\n')        
        f.write("predicted- buggy: "+str(Counter(predict1D)[1])+" non-buggy: "+str(Counter(predict1D)[0])+'\n')

    cm = confusion_matrix(actual1D, predict1D)
    actual_class = Counter(actual1D)
    with open(out_file_name, 'a') as f:
        f.write("correct- total: {} ({:0.2f}%), buggy: {} ({:0.2f}%) non-buggy: {} ({:0.2f}%)\n".format(
            cm[0][0] + cm[1][1],
            ((cm[0][0] + cm[1][1])/len(actual1D))*100, 
            cm[1][1],
            (cm[1][1]/actual_class[1])*100, 
            cm[0][0], 
            (cm[0][0]/actual_class[0])*100))

    with open(out_file_name, 'a') as f:
        f.write("performance: "+str(round(result[0], 2))+", "+str(round(result[1], 2))+", "+str(round(result[2], 2))+'\n')
        