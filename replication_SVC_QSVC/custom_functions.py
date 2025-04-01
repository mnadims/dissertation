from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, matthews_corrcoef, precision_recall_curve
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from sklearn.feature_selection import VarianceThreshold
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from sklearn.utils import resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from sklearn.utils import resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
#------------------------------------------------------
def select_n_features(df, n):
    #n = 10
    # Separate features and target class
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target class

    # Apply SelectKBest class to extract top n best features
    best_features = SelectKBest(score_func=mutual_info_classif, k=n)
    fit = best_features.fit(X, y)

    # Get selected features
    selected_features = fit.transform(X)
    
    selected_feature_names = X.columns[fit.get_support()]

    # Create a DataFrame with selected features
    selected_df = pd.DataFrame(selected_features)

    # Concatenate the target class to the selected features DataFrame
    selected_df[df.columns[-1]] = y.values
    print(selected_feature_names)
    return selected_df
#------------------------------------------------------
def my_resample(df):
    # Upsample the True class
    df_true = df[df.iloc[:, -1] == True]

    # Downsample the False class
    df_false = df[df.iloc[:, -1] == False].sample(n=len(df_true), replace=False)

    # Concatenate the upsampled True class with the downsampled False class
    df_resampled = pd.concat([df_true, df_false])
    
    # Shuffle the DataFrame to randomize the order
    df_resampled = df_resampled.sample(frac=1).reset_index(drop=True)
    
    return df_resampled
#------------------------------------------------------
def find_top_index(results, n=5, metric=4, show_result=False, sort_values=True):
    if sort_values:
        sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
    else: 
        sorted_results = results
    
    if show_result:
        for sr in sorted_results:
            print(sr)
    top_index = [r[0] for r in sorted_results[:n]]
    return top_index
#------------------------------------------------------
def read_prediction(filename):
    testY = open(filename).read().replace(' ', '').replace("'", "").split(',')
    testY = [int(x) for x in testY[:-1]]
    return testY    
#------------------------------------------------------
def inc_prediction(clf, testX):
    total_iterations = len(testX)
    with tqdm(total=total_iterations) as pbar: #creates a progress bar!
        n=1
        predictions = []
        st = time.time()      
        while n <=total_iterations:          
            p = clf.predict(testX.iloc[n-1:n, :])             
            predictions.append(p[0])     
            
            if n%10 == 0:
                pbar.update(10) #update once in 10 iterations!
            n+=1

        et = time.time()  
        
    return predictions, round(et - st, 2) 
#------------------------------------------------------
def remove_constant_columns(X_train):
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_train)
    #constant_columns = [column for column in X_train.columns if column not in X_train.columns[constant_filter.get_support()]]
    surviving_columns = list(X_train.columns[constant_filter.get_support()])
    X_train = constant_filter.transform(X_train)    
    return pd.DataFrame(X_train, columns=surviving_columns)
#------------------------------------------------------
def getNBestFeatures(df, n):
    y = df['contains_bug'].astype(int)
    X = df.drop(columns=['contains_bug'])
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
#------------------------------------------------------
def split_train_test_NBest(df, n):
    n_best_f = getNBestFeatures(df, n)
    ds = df[["contains_bug"]+n_best_f]

    buggy_ds = ds[ds["contains_bug"] == 1]
    clean_ds = ds[ds["contains_bug"] == 0]

    #print(buggy_ds.shape)

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

    #print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY
#------------------------------------------------------
def read_df(file, split=True, sample_size=1):
    df = pd.read_csv(file, low_memory=False)
    df.replace('?', np.nan, inplace=True)
    #df.fillna(df.mean(), inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.sample(frac=sample_size).reset_index(drop=True) 
    
    if split: 
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1].astype('int')
        return X, Y
    else:
        return df
#------------------------------------------------------
def print_result(testY, predictY):
    # Accuracy
    accuracy = accuracy_score(testY, predictY)

    # Precision
    precision = precision_score(testY, predictY)

    # Recall
    recall = recall_score(testY, predictY)

    # F1-score
    f1 = f1_score(testY, predictY)

    # ROC AUC
    roc_auc = roc_auc_score(testY, predictY)

    # Confusion matrix
    conf_matrix = confusion_matrix(testY, predictY)

    # Classification report
    class_report = classification_report(testY, predictY)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(testY, predictY)
    
    return [round(x, 2) for x in [accuracy, precision, recall, f1, roc_auc, mcc]]

'''    
    # Output results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC AUC:", roc_auc)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)
    print("Matthews Correlation Coefficient:", mcc)
    print("----------------------")
'''
#------------------------------------------------------
def my_round(x, flag = 0.50):    
    if x>=flag:
        return 1 
    else:
        return 0
#------------------------------------------------------
def aggregate_pred(list_predictions, flag=0.50):
    return [my_round(x, flag) for x in np.mean(np.array(list_predictions), axis=0)]

#------------------------------------------------------
def getQAlgo(algo):
    seed = 1376
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
        return PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)   
    else: 
        return QSVC(quantum_kernel=qkernel)    
    
#------------------------------------------------------
def weighted_average(arr, weights):
    # Ensure the array and weights have compatible shapes
    if arr.shape[0] != len(weights):
        raise ValueError("Number of rows in the array must match the length of weights")

    # Expand dimensions of weights to allow broadcasting
    weights = np.expand_dims(weights, axis=1)

    # Calculate weighted average
    weighted_avg = np.sum(arr * weights, axis=0) / np.sum(weights)

    return weighted_avg
#------------------------------------------------------
def run_classical(clf, trainX, trainY, testX, testY):
    classic_p = clf.fit(trainX, trainY).predict(testX)
    print("Confusion Matrix:\n", confusion_matrix(testY, classic_p))
    print(print_result(testY, classic_p))
#------------------------------------------------------
def save_to_file(p_arr, outfile="output/test_dump/temp.txt"):
    out_text = ''
    for p in p_arr:
        out_text += str(p)+', '
        
    with open(outfile, 'w') as f:
        f.write(out_text)
#------------------------------------------------------
def print_arr(p_arr, name= ""):
    print(name, end='')
    
    for p in p_arr:
        print(p, end=', ')
        out_text += p+', '
    print()
#------------------------------------------------------
def calculate_cbcr(precision, recall, total_buggy_instances):
    true_positives = precision * recall * total_buggy_instances
    false_negatives = total_buggy_instances - true_positives
    cbcr = true_positives / (true_positives + false_negatives)
    return cbcr

#------------------------------------------------------
def plot_prediction(filename, testY, prediction_list):
    precision, recall, thresholds = precision_recall_curve(testY, np.mean(np.array(prediction_list), axis=0))
    fscores = [2*(r*p)/(r+p) for r, p in zip(recall, precision)]

    data_exp = zip(thresholds, precision, recall, fscores)
    list_data = []
    for d in data_exp:
        list_data.append(d)
        #print(d)

    sorted_data = sorted(list_data, key=lambda x: x[3], reverse=True)

    fig, ax1 = plt.subplots()
    ax1.plot(precision, recall, marker='.', linestyle='-', label='Precision-Recall Curve')
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Recall')
    
    # Create a second x-axis on the top
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    plt_th = [round(th, 2) for th in thresholds]
    #print(thresholds)
    ax2.set_xticks(plt_th)
    ax2.set_xlabel('Candidate Thresholds')
    ax2.tick_params(axis='x', rotation=30)

    for sd in sorted_data[:3]:
        if sd[0] > 0 and sd[0] <= 1:
            ax1.annotate(f'{sd[0]:.2f}, {sd[1]:.2f}, {sd[2]:.2f}, {sd[3]:.2f}', (sd[1], sd[2]))

    # # Annotate each point with its corresponding threshold value
    # for i, threshold in enumerate(thresholds):
    #     if threshold>0 and threshold<1:
    #         plt.annotate(f'{threshold:.2f}, {precision[i]:.2f}, {recall[i]:.2f}, {fscores[i]:.2f}', (precision[i], recall[i]))

   
    #plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.5)
    ax1.legend(loc ='lower left')
    plt.savefig(f'inc_saved_images/{filename}.png') #Save the image to disk
    plt.show()
#------------------------------------------------------

#------------------------------------------------------