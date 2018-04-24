import os
import sys
import pickle
import csv
import numpy as np
import essentia.standard as ess
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import warnings
warnings.filterwarnings('ignore')
import IPython.display as ipd
import json

from pandas import DataFrame, read_csv
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score

######## ARGUMENT PARSING ########


####### DOWNLOAD DATASET #########



####### DATA STRUCTURES ##########

#Container for analysis parameters
class AnalysisParams:
    def __init__(self,windowSize,hopSize,windowFunction,fftN,fs,numBins):
        '''
        windowSize: milliseconds,
        hopSize: milliseconds,
        windowFunction: str ('blackman','hanning',...)
        fftN: int
        '''
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.windowFunction = windowFunction
        self.fftN=fftN
        self.fs=fs
        self.numBins = numBins

def initiateData4File(file,root):
    '''Forming the data structure for file
    Parameters
    ----------
    file,root : str
        File name and path info
    
    Returns
    -------
    fileData : dict
        Dictionary containing all info and data for the file
    ''' 
    fileData=dict();fileData['name']=file.split('.')[0];fileData['path']=root;
    fileData['numBins']=[];
    fileData['hpcp']=[];
    fileData['mean_hpcp_vector']=[];
    fileData['std_hpcp_vector']=[];
    #data from annotations
    fileData['groundtruth']=[];
    fileData['key']=[];
    fileData['tuning']=[];
    
    return fileData

def sliceAudiotoParts(audio,endTime,startTime,params): #slicing the audio into parts according to annotated timestamps
    fs = params.fs
    endtime=float(endTime)
    starttime=float(startTime)
    audio_slice = audio[starttime*fs:endtime*fs]
    return audio_slice

def createDataStructure(targetDir,numBins):
    dataDict = dict()
    
    for root, dirs, files in os.walk(targetDir):
        for file in files:
            if file.endswith('.json'):
                with open(targetDir+file) as json_file:
                    annotationData=json.load(json_file)
                    fileName=file.split('.')[0]
                    dataParts=annotationData['parts']
                    for i in range (len(dataParts)):
                    
                        fileData=initiateData4File(file,root)            
                        files4scale=dataDict.get(fileName)
                        if files4scale==None:
                            files4scale=[fileData]
                        else:
                            files4scale.append(fileData)
                    
                        dataDict[fileName]=files4scale
                        dataDict[fileName][i]['groundtruth']=dataParts[i]
                        dataDict[fileName][i]['key']=annotationData['sandbox']['key']                        
                        dataDict[fileName][i]['numBins'] = numBins
    return dataDict
                        

######### FEATURE EXTRACTION #################


def computeReferenceFrequency(tonic,tuningfreq):    #computation of the reference frequency for HPCP vector from the tonic of the audio segment
    keys = {'A':0,'Bb':1,'B':2,'C':3,'C#':4,'D':5,'Eb':6,'E':7,'F':8,'F#':9,'G':10,'Ab':11}
    for key in keys:
        if key == tonic:
            ref_freq = tuningfreq * (2**((keys[key]*100)/1200)) /2 #compute reference frequency according to the key (tonic) of the audio file
                                                     #divide the results by 2 the get one octave lower pitch as the reference freq
            return ref_freq
        
def computeHPCP(x,windowSize,hopSize,params,fileData):   
    
    #Initializing lists for features
    hpcp=[];
    numBins = params.numBins
    
    #Main windowing and feature extraction loop
    for frame in ess.FrameGenerator(x, frameSize=windowSize, hopSize=hopSize, startFromZero=True):
        frame=ess.Windowing(size=windowSize, type=params.windowFunction)(frame)
        mX = ess.Spectrum(size=windowSize)(frame)
        mX[mX<np.finfo(float).eps]=np.finfo(float).eps
        
        freq,mag = ess.SpectralPeaks()(mX) #extract frequency and magnitude information by finding the spectral peaks
        tunefreq, tunecents = ess.TuningFrequency()(freq,mag)
        reffreq = computeReferenceFrequency(fileData['key'][0],tunefreq)
        hpcp.append(ess.HPCP(normalized='unitSum',referenceFrequency=reffreq,size = numBins, windowSize = 12/numBins)(freq,mag)) #harmonic pitch-class profiles 
        
    return hpcp, tunefreq

def computeHPCPFeatures(fileData,params,numBins):
    '''Computation of the low-level features
    Parameters
    ----------
    fileData : dict
        Dictionary containing all info and data for the file
    params : instance of AnalysisParams
        Analysis parameters
    Modifies
    -------
    fileData 
    '''
    ###################################### TODO - REFACTOR CODE ######################3
    #Reading the wave file
    fs=params.fs
    x = ess.MonoLoader(filename = os.path.join('audio/', fileData['name']+'.mp3'), sampleRate = fs)()
    x = ess.DCRemoval()(x) ##preprocessing / apply DC removal for noisy regions
    x = ess.EqualLoudness()(x)
    #Windowing (first converting from msec to number of samples)
    windowSize=round(fs*params.windowSize/1000);windowSize=int(windowSize/2)*2#assuring window size is even
    hopSize=round(fs*params.hopSize/1000);hopSize=int(hopSize/2)*2#assuring hopSize is even
    
    #slicing audio
    startTime=fileData['groundtruth']['startTime']
    endTime=fileData['groundtruth']['endTime']
    x_slice = sliceAudiotoParts(x,endTime,startTime,params)
    
    HPCPs, tuningfreq = computeHPCP(x_slice,windowSize,hopSize,params,fileData)
    fileData['hpcp']=np.array(HPCPs);
    fileData['tuning'] = tuningfreq;
    
def computeGlobHPCP(fileData):
    '''Computation of the global features from low-level features
   
    Parameters
    ----------
    fileData : dict
        Dictionary containing all info and data for the file
    Modifies
    -------
    fileData 
    '''
    
    features=list(fileData.keys())
    features.remove('path');features.remove('name')
    
    for j in range(fileData['numBins']):
        hpcps = [];
        for i in range(len(fileData['hpcp'])):
            hpcps.append(fileData['hpcp'][i][j])
        fileData['mean_hpcp_vector'].append(np.mean(hpcps))
        fileData['std_hpcp_vector'].append(np.std(hpcps))   
                
############ DATA FORMATTING ###################

def generateCSV(filename, dataDir):
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    #numBins = data['toprak_dorian'][0]['numBins']    
    numBins = 12
    fieldnames=['name']
    for i in range(numBins):
        ind=str(i)
        fieldnames.append('mean_hpcp'+ind)
    for i in range(numBins):
        ind=str(i)
        fieldnames.append('std_hpcp'+ind)    
    fieldnames.append('scaleType')
    dataList=[]
    dataList.append(fieldnames)
    for fileName, parts in data.items(): ##navigate in dictionary
        for part in parts: #search within audio slices
            tempList=[] #temporary List to put attributes for each audio slice (data-point)
            dataname = part['name']+'_'+part['groundtruth']['name'] #name of data
            tempList.append(dataname)
            for i in range(numBins): #append mean_HPCP vector bins separately            
                tempList.append(part['mean_hpcp_vector'][i])
            for i in range(numBins): #append mean_HPCP vector bins separately            
                tempList.append(part['std_hpcp_vector'][i])    
            tempList.append(part['groundtruth']['scaleType'].split(':')[1])    #append scales for classification
            dataList.append(tempList)

    with open(dataDir+'CSVfilefor_'+str(numBins)+'bins.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataList)    
        
############# MACHINE LEARNING ####################

def Classification(filename,dataDir):
      
        
    numBins = filename.split('_')[1].split('bins')[0]
      
     
    
    print('This process might take a while (5-10 min) \n CROSS-VALIDATION & TRAINING ') 
    list_accuracy=[]
    
    df = pd.read_csv(os.path.join(dataDir,filename))
    df.pop('name'); dataclass=df.pop('scaleType')
    X=df; Y=dataclass
    modeSet = set(Y)
    
    cm,acc,f = machineLearningEvaluation(dataDir,X,Y,numBins)

    modeSet = sorted(modeSet)
    print(modeSet)
    plot_confusion_matrix(cm,modeSet,normalize=False)
    

def hyperparameterOptimization(X_train, Y_train, nfolds):
    
    ##HYPERPARAMETER OPTIMIZATION USING GRID SEARCH WITH 10-fold CROSS VALIDATION

    ##HYPERPARAMETER SET:
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    ## APPLY CROSS N-FOLD CROSS VALIDATION, ITERATING OVER PARAMETER GRIDS

    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    ## RETURN PARAMETERS WITH MOST ACCURACATE CROSS-VALIDATION SCORES
    return grid_search.best_params_

def machineLearningEvaluation(targetDir, X, Y, numBin):
    f_measures = []
    accuracies = []
    Y_total = [];
    Y_pred_total = [];
    ## TO INCREASE GENERALIZATION POWER, THE TRAIN-VALIDATE-TEST PROCEDURE IS PERFORMED
    ## OVER 10 RANDOM INSTANCES.
    for randomseed in range(30, 50, 2):
        ## SPLIT DATASET INTO TRAIN_SET & TEST_SET, IMPORTANT ---> STRATIFY
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=randomseed)

        ##OPTIMIZE PARAMETERS ON TRAIN SET.
        ##IMPORTANT --> TEST SET CANNOT INFLUENCE THE MODEL. IF SO -> RISK OF OVERFITTING
        param = hyperparameterOptimization(X_train, Y_train, 10)

        ## CREATE A PIPELINE
        estimators = []
        ##CREATE THE MODEL WITH OPTIMIZED PARAMETERS
        model1 = SVC(C=param['C'], gamma=param['gamma'])
        estimators.append(('classify', model1))
        model = Pipeline(estimators)
        ## TRAIN MODEL WITH TRAINING SET & PREDICT USING X_TEST_FEATURES
        Y_pred = model.fit(X_train, Y_train).predict(X_test)

        ##EVALUATION
        ## TEST PREDICTED VALUES WITH GROUND TRUTH (Y_TEST)
        accscore = accuracy_score(Y_test, Y_pred)
        #print('Accuracy Measure = ')
        #print(accscore)

        f_measure = f1_score(Y_test, Y_pred, average='weighted')
        #print('F Measure = ')
        #print(f_measure)

        f_measures.append(f_measure)
        accuracies.append(accscore)
        for i in Y_test:
            Y_total.append(i)
        for i in Y_pred:
            Y_pred_total.append(i)

    print('Accuracy score for the Feature Set : ')
    
    ##AVERAGE ALL RANDOM SEED ITERATIONS FOR GENERALIZATION
    print('F-measure (mean,std) --- FINAL')
    f = round(np.mean(f_measures) ,2)
    fstd = np.std(f_measures)
    print(f,fstd )
    print('Accuracy (mean,std) FINAL')
    ac = round(np.mean(accuracies), 2)
    accstd=np.std(accuracies)
    print(ac,accstd)
    cm = confusion_matrix(Y_total, Y_pred_total)


    with open(targetDir + 'scores_' + str(numBin) + '.txt', 'w') as scorefile:
        scorefile.write(str(f))
        scorefile.write(str(ac))
        
    return cm, f_measures, accuracies

############ PLOTTING ################

def plot_confusion_matrix(cm, classes,
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
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.figure(figsize=(9, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title, fontsize=26)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tick_params(labelsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", size=20)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=24)
    plt.xlabel('Predicted label', fontsize=24)
    plt.show()

#######  SINGLE FILE PROCESSING ###########

   
def FeatureExtraction_single(fileName, fileDir, params,annotationFile):
    '''
    fileName : input audio file (.mp3)
    fileDir : directory of the audio file
    params : Analysis parameters object
    annotationData : Data from the annotation (exercise.json)
    '''
    
    numBins = params.numBins    
    fs=params.fs
    
    x = ess.MonoLoader(filename = os.path.join(fileDir, fileName), sampleRate = fs)()
    x = ess.DCRemoval()(x) ##preprocessing / apply DC removal for noisy regions
    x = ess.EqualLoudness()(x)
    #Windowing (first converting from msec to number of samples)
    windowSize=round(fs*params.windowSize/1000);windowSize=int(windowSize/2)*2#assuring window size is even
    hopSize=round(fs*params.hopSize/1000);hopSize=int(hopSize/2)*2#assuring hopSize is even
    
    with open(fileDir+annotationFile) as json_file:
        annotationData=json.load(json_file)
    partsList = dict()  
    
    for part in annotationData['parts']:
        fileData=initiateData4File(part['name'],fileDir)
        fileData['startTime'] = part['startTime']
        fileData['endTime'] = part['endTime']
        fileData['key'] = part['scaleType'].split(':')[0]
        fileData['scaleType'] = part['scaleType'].split(':')[1]
        
        x_slice = sliceAudiotoParts(x,fileData['endTime'],fileData['startTime'],params)
        hpcp = []
        for frame in ess.FrameGenerator(x_slice, frameSize=windowSize, hopSize=hopSize, startFromZero=True):
            frame=ess.Windowing(size=windowSize, type=params.windowFunction)(frame)
            mX = ess.Spectrum(size=windowSize)(frame)
            mX[mX<np.finfo(float).eps]=np.finfo(float).eps

            freq,mag = ess.SpectralPeaks()(mX) #extract frequency and magnitude information by finding the spectral peaks
            tunefreq, tunecents = ess.TuningFrequency()(freq,mag)
            reffreq = computeReferenceFrequency(fileData['key'],tunefreq)
            hpcp.append(ess.HPCP(normalized='unitSum',referenceFrequency=reffreq,size = numBins, windowSize = 12/numBins)(freq,mag))
            
        for j in range(numBins):
            hpcps = [];
            for i in range(len(hpcp)):
                hpcps.append(hpcp[i][j])
            fileData['mean_hpcp_vector'].append(np.mean(hpcps))
            fileData['std_hpcp_vector'].append(np.std(hpcps))
            
        partsList[part['name']] = fileData
        
        
    fieldnames=['name']
    for i in range(numBins):
        ind=str(i)
        fieldnames.append('mean_hpcp'+ind)
    for i in range(numBins):
        ind=str(i)
        fieldnames.append('std_hpcp'+ind)    
    fieldnames.append('scaleType')
    
    dataList=[]
    dataList.append(fieldnames)
    
    for part in partsList.items(): ##navigate in dictionary
         
        tempList=[] #temporary List to put attributes for each audio slice (data-point)
        
        dataname = part[0] #name of data
        tempList.append(dataname)
        for i in range(numBins): #append mean_HPCP vector bins separately               
            tempList.append(part[1]['mean_hpcp_vector'][i])
        for i in range(numBins): #append mean_HPCP vector bins separately            
            tempList.append(part[1]['std_hpcp_vector'][i])    
        tempList.append(part[1]['scaleType'])    #append scales for classification
        dataList.append(tempList)

    dataList = sorted(dataList, key=lambda x: x[0])
    
    dataListSorted = []
    dataListSorted.append(dataList[-1])
    for i in range(len(dataList)-1):
        dataListSorted.append(dataList[i])
    
    with open(fileDir+'CSVfilefor_'+str(numBins)+'bins.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataListSorted)    
        
    return partsList

def TrainANDPredict(filenameTRAIN,filenamePREDICT,dataDir):
    
    dfTRAIN = pd.read_csv(os.path.join(dataDir,filenameTRAIN))
    dfTRAIN.pop('name'); dataclass=dfTRAIN.pop('scaleType')
    X=dfTRAIN; Y=dataclass
    modeSet = set(Y)
    
    dfPREDICT = pd.read_csv(os.path.join(dataDir,filenamePREDICT))
    dfPREDICT.pop('name'); dataclassPREDICT=dfPREDICT.pop('scaleType')
    X_PREDICT=dfPREDICT; Y_PREDICT=dataclassPREDICT
    
    
    param = hyperparameterOptimization(X, Y, 10)
    
    ## CREATE A PIPELINE
    estimators = []
    ##CREATE THE MODEL WITH OPTIMIZED PARAMETERS
    model1 = SVC(C=param['C'], gamma=param['gamma'])
    estimators.append(('classify', model1))
    model = Pipeline(estimators)
    ## TRAIN MODEL WITH TRAINING SET & PREDICT USING X_TEST_FEATURES
    return(model.fit(X, Y).predict(X_PREDICT))
    
#def PredictScaleType(partsList, trainedModel):
    '''
    for part in partsList:
        X_test = []
        for j in range(
        X_test = part
    
    '''
    
    