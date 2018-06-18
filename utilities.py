import os, sys, csv, pickle, itertools, warnings, json, math
import numpy as np
import essentia.standard as ess
import matplotlib.pyplot as plt

import essentia
essentia.log.warningActive=False
warnings.filterwarnings('ignore')
import IPython.display as ipd
import pandas as pd
from pandas import DataFrame, read_csv
import seaborn as sn

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score
from sklearn.preprocessing import normalize


######## ARGUMENT PARSING ########


####### DOWNLOAD DATASET #########

####### SCALE DICTIONARY ########

def ScaleDictionary():
    
    ScaleTemplates = dict()
    
    ScaleTemplates['major'] = {'scaleArray':[1,0,1,0,1,1,0,1,0,1,0,1]}
    ScaleTemplates['dorian'] = {'scaleArray':[1,0,1,1,0,1,0,1,0,1,1,0]}
    ScaleTemplates['phrygian'] = {'scaleArray':[1,1,0,1,0,1,0,1,1,0,1,0]}
    ScaleTemplates['lydian'] = {'scaleArray':[1,0,1,0,1,0,1,1,0,1,0,1]}
    ScaleTemplates['mixolydian'] = {'scaleArray':[1,0,1,0,1,1,0,1,0,1,1,0]}
    ScaleTemplates['minor'] = {'scaleArray':[1,0,1,1,0,1,0,1,1,0,1,0]}
    ScaleTemplates['locrian'] = {'scaleArray':[1,1,0,1,0,1,1,0,1,0,1,0]}
    ScaleTemplates['lydianb7'] = {'scaleArray':[1,0,1,0,1,0,1,1,0,1,1,0]}
    ScaleTemplates['altered'] = {'scaleArray':[1,1,0,1,1,0,1,0,1,0,1,0]}
    ScaleTemplates['melmin'] = {'scaleArray':[1,0,1,1,0,1,0,1,0,1,0,1]}
    ScaleTemplates['hminor'] = {'scaleArray':[1,0,1,1,0,1,0,1,1,0,0,1]}
    ScaleTemplates['wholetone'] = {'scaleArray':[1,0,1,0,1,0,1,0,1,0,1,0]}
    ScaleTemplates['hwdiminished'] = {'scaleArray':[1,1,0,1,1,0,1,1,0,1,1,0]}
    
    return ScaleTemplates

####### DISPLAY DATASET #########

def get_sound_embed_html(freesound_id):
    return '<iframe frameborder="0" scrolling="no" src="http://www.freesound.org/embed/sound/iframe/%i/simple/medium/" width="481" height="86"></iframe>' % freesound_id

def generate_html_with_sound_examples(freesound_ids):
    html = ''
    for sound_id in freesound_ids:
        html += get_sound_embed_html(sound_id)
    return html

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
    fileData=dict(); fileData['name']=file.split('.')[0]; fileData['path']=root;
    fileData['numBins']=[]; fileData['hpcp']=[]; fileData['mean_hpcp_vector']=[]; fileData['std_hpcp_vector']=[];
    #data from annotations
    fileData['groundtruth']=[]; fileData['key']=[]; fileData['tuning']=[];
    
    return fileData

def sliceAudiotoParts(audio,endTime,startTime,params): #slicing the audio into parts according to annotated timestamps
    fs = params.fs
    endtime=float(endTime)
    starttime=float(startTime)
    audio_slice = audio[starttime*fs:endtime*fs]
    return audio_slice

def createDataStructure(targetDir):
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
                        
    return dataDict
                        
######### FEATURE EXTRACTION #################

def computeReferenceFrequency(tonic,tuningfreq):    #computation of the reference frequency for HPCP vector from the tonic of the audio segment
    keys = {'A':0,'Bb':1,'B':2,'C':3,'C#':4,'D':5,'Eb':6,'E':7,'F':8,'F#':9,'G':10,'Ab':11}
    for key in keys:
        if key == tonic:            
            #compute reference frequency according to the key (tonic) of the audio file
            #divide the results by 2 the get one octave lower pitch as the reference freq
            return (tuningfreq * (2**((keys[key]*100)/1200)) /2)                         
        
def computeHPCPFeatures(x,windowSize,hopSize,params,fileData):   
    
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

def computeHPCP_FRAMEBASED(fileData,params):
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
    #x = ess.DCRemoval()(x) ##preprocessing / apply DC removal for noisy regions
    x = ess.EqualLoudness()(x)
    #Windowing (first converting from msec to number of samples)
    windowSize=round(fs*params.windowSize/1000);windowSize=int(windowSize/2)*2#assuring window size is even
    hopSize=round(fs*params.hopSize/1000);hopSize=int(hopSize/2)*2#assuring hopSize is even
    
    #slicing audio
    startTime=fileData['groundtruth']['startTime']
    endTime=fileData['groundtruth']['endTime']
    x_slice = sliceAudiotoParts(x,endTime,startTime,params)
    
    HPCPs, tuningfreq = computeHPCPFeatures(x_slice,windowSize,hopSize,params,fileData)
    
    for i in range(len(HPCPs)):
        dummy = np.zeros_like(HPCPs[i])
        dummy[HPCPs[i].argmax(0)] = HPCPs[i].max(0)
        HPCPs[i] = dummy
    
    fileData['hpcp']=np.array(HPCPs);
    fileData['tuning'] = tuningfreq;
    
def computeHPCP_GLOBAL(fileData):
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
    
    hpcpsMEAN = []
    hpcpsSTD = []
    for j in range(len(fileData['hpcp'][0])):
        hpcpBIN = [];
        for i in range(len(fileData['hpcp'])):
            hpcpBIN.append(fileData['hpcp'][i][j])
        hpcpsMEAN.append(np.mean(hpcpBIN)) 
        hpcpsSTD.append(np.std(hpcpBIN))
    
    hpcpsMEANSUM = np.sum(hpcpsMEAN)
    for i in range(len(hpcpsMEAN)):
        hpcpsMEAN[i] = hpcpsMEAN[i] / hpcpsMEANSUM
    fileData['mean_hpcp_vector'] = hpcpsMEAN
    
    hpcpsSUM = np.sum(hpcpsSTD)
    for i in range(len(hpcpsSTD)):
        hpcpsSTD[i] = hpcpsSTD[i] / hpcpsSUM    
    fileData['std_hpcp_vector'] =  hpcpsSTD
    
############ DATA FORMATTING ###################

def FeatureSelection(filename, dataDir, featureSet):
    
    ''' 
    featureSet = 1 for ONLY mean HPCP, 2 for ONLY std HPCP, 3 for BOTH mean + std HPCP
    '''
    
    with open(os.path.join(dataDir,filename), 'rb') as f:
        data = pickle.load(f)
    #numBins = data['toprak_dorian'][0]['numBins']    
    numBins = 12
    fieldnames=['name']
    
    if featureSet == 1 or featureSet == 3:
        for i in range(numBins):
            ind=str(i)
            fieldnames.append('mean_hpcp'+ind)
    if featureSet == 2 or featureSet == 3:        
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
            
            if featureSet == 1 or featureSet == 3:
                for i in range(numBins): #append mean_HPCP vector bins separately            
                    tempList.append(part['mean_hpcp_vector'][i])
            if featureSet == 2 or featureSet == 3:        
                for i in range(numBins): #append mean_HPCP vector bins separately            
                    tempList.append(part['std_hpcp_vector'][i])    
            tempList.append(part['groundtruth']['scaleType'].split(':')[1])    #append scales for classification
                
            dataList.append(tempList)

    with open(dataDir+'FeaturesData_ChordScaleDataset.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataList)    
        
############# VISUALIZATION - CHROMA HISTOGRAM #################

def plotChromaHistograms(fileData, scaleTemplate):
    
    ChromaMEAN = fileData['mean_hpcp_vector']
    ChromaSTD = fileData['std_hpcp_vector']
    
    x1 = np.multiply(ChromaMEAN,scaleTemplate)
    x2 = np.multiply(ChromaMEAN, np.abs(1-np.array(scaleTemplate)))
   
    fig, axes = plt.subplots(1, 2, sharey=True, tight_layout=True, figsize = (10,6))
       
    axes[0].bar(np.arange(len(x1)),x1, label = 'In-Scale Notes')
    axes[0].bar(np.arange(len(x2)),x2, color = 'coral', label = 'Out-Scale Notes')
    axes[0].set_title('CHROMA HISTOGRAM (MEAN)')
    axes[0].set_xlabel('PITCH CLASSES')
    axes[0].set_ylabel('AMPLITUDE (NORMALIZED)')
    axes[0].legend(['In-Scale Notes','Out-Scale Notes'], loc="upper right")
    
    fig.suptitle(fileData['name'] + '\n Scale Type : ' + fileData['groundtruth']['scaleType'].split(':')[1],fontsize=16)
    
    x3 = np.multiply(ChromaSTD,scaleTemplate)
    x4 = np.multiply(ChromaSTD, np.abs(1-np.array(scaleTemplate)))

    axes[1].bar(np.arange(len(x3)),x3, color = 'teal',label = 'In-Scale Notes')
    axes[1].bar(np.arange(len(x4)),x4, color = 'coral', label = 'Out-Scale Notes')
    axes[1].set_title('CHROMA HISTOGRAM (STD)')
    axes[1].set_xlabel('PITCH CLASSES')
    axes[1].set_ylabel('AMPLITUDE (NORMALIZED)')
    axes[1].legend(['In-Scale Notes','Out-Scale Notes'], loc="upper right")
    
    plt.subplots_adjust(left=0.35, right = 0.7, wspace=0.6, top=0.05, bottom = 0.02)
    
    plt.show()
            
############ CHORD - SCALE DETECTION - METHOD 1: TEMPLATE-BASED LIKELIHOOD ESTIMATION ##############


def ScaleLikelihoodEstimation(ChromaVector, ScaleTemplates, method):   
    
    scalesList = ['major','dorian','phrygian','lydian','mixolydian','minor','locrian','melmin','lydianb7','altered','hminor', 'hwdiminished']
    
    
    likelihoods = []
    likelihoodvalues = []
      
    for i in range(len(scalesList)):
        
        if method == 1:
        
            Likelihood = Maxlikelihood_MULTIPLICATION(ChromaVector,ScaleTemplates[scalesList[i]]['scaleArray'])
        
        elif method == 2:
    
            Likelihood = Maxlikelihood_SUMMATION(ChromaVector,ScaleTemplates[scalesList[i]]['scaleArray'])
                    
        likelihoodvalues.append(Likelihood)
        
    likelihoodvaluesSUM = np.sum(likelihoodvalues)    
    likelihoodsNormalized = []
    for i in range(len(likelihoodvalues)):
        likelihoodsNormalized.append(likelihoodvalues[i]/likelihoodvaluesSUM)
    
    
    for i in range(len(scalesList)):
        
        likelihoods.append((scalesList[i],likelihoodsNormalized[i]))
        
    maxLikelihood = [(scalesList[np.argmax(likelihoodsNormalized)],np.max(likelihoodsNormalized))]

    return(maxLikelihood,likelihoods)

def Maxlikelihood_MULTIPLICATION(ChromaVector,scaleArray):
    
    NumNotesScale = np.sum(scaleArray)
    ChromaScaled = []
    for i in range(len(ChromaVector)):
        ChromaScaled.append(ChromaVector[i]**scaleArray[i])
           
    #ChromaScaled = np.power(ChromaVector,scaleArray)    
    scale_likelihood = np.prod(ChromaScaled) / ((1/NumNotesScale)**NumNotesScale)

    return(scale_likelihood) 
    
def Maxlikelihood_SUMMATION(ChromaVector,scaleArray):
    
    NumNotesScale = np.sum(scaleArray)
    ChromaScaled = np.multiply(ChromaVector,scaleArray)        
    scale_likelihood = np.sum(ChromaScaled) / NumNotesScale        

    return(scale_likelihood)

############ VISUALIZATION - SCALE LIKELIHOODS ########################

def VisualizeChromaANDScaleLikelihoods(FileData, scaleTemplates, likelihoodmethod, features):

    
    if features == 'HPCP':
        hpcps = FileData['hpcp']
    elif features == 'NNLS':
        hpcps = FileData['NNLS']
    hpcpAgg = np.zeros_like(hpcps[0])

    scalelikelihoods = []
    
    for i in range(len(hpcps)):
        hpcpAgg = hpcpAgg + hpcps[i]
        #print(hpcpAgg)
        maxscalelike, likelihood = ScaleLikelihoodEstimation(hpcpAgg,scaleTemplates, likelihoodmethod)        
        
        framelikelihoods = []
        
        for l in range(len(likelihood)):
            framelikelihoods.append(likelihood[l][1])
            
        scalelikelihoods.append(framelikelihoods)
        
    
    pitch_classes = ['A','Bb','B','C','C#','D','D#','E','F','F#','G','G#']

    
    #### Order of scale Types : 
    
    scaleTypes = ['major','dorian','phrygian','lydian','mixolydian','minor','locrian','melmin','lydianb7','altered','hminor', 'hwdiminished']
                   
    MaxLikelihoodNormalized = maxscalelike[0][1] / np.sum(scalelikelihoods[-1])
    print('Maximum Likeliest Scale of Phrase : ' + str(maxscalelike[0][0]) )
    #print(scalelikelihoods)
    '''
    for i in range(len(scalelikelihoods)):
        scalelikelihoodsSUM = np.sum(scalelikelihoods[i])
        for j in range(len(scalelikelihoods[i])):
            scalelikelihoods[i][j] = scalelikelihoods[i][j] / scalelikelihoodsSUM
    '''
       
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(10,6), tight_layout=True)
    
    ax1.imshow(np.transpose(scalelikelihoods),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma')
    ax1.set_xlabel('Frame #')
    ax1.set_ylabel('ScaleTypes')
    tick_marks = np.arange(len(scaleTypes))
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(scaleTypes)
           
    ax2.imshow(np.transpose(hpcps),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma')
    ax2.set_xlabel('Frame #')
    ax2.set_ylabel('Pitch-Classes')
    tick_marks1 = np.arange(len(pitch_classes))
    ax2.set_yticks(tick_marks1)
    
    plt.subplots_adjust(left=0.6, right = 0.7, wspace=0.2, top=0.65, bottom = 0.6)
    
    plt.show()
        
def Classification_Likelihood(dataDictionary, ScaleTemplates, likelihoodmethod, features, featureSet):
        
    CORRECT_SCALE = 0
    ALL_PARTS = 0
    
    if features == 'HPCP':
        if featureSet =='mean':
            features = 'mean_hpcp_vector'
        elif featureSet =='std':   
            features = 'std_hpcp_vector'
    
    elif features == 'NNLS':
        if featureSet =='mean':
            features = 'mean_NNLS'
        elif featureSet =='std':   
            features = 'std_NNLS'

    for files, parts in dataDictionary.items():
        for part in parts:
            LikeliestScale, ScaleLikelihoods = ScaleLikelihoodEstimation(part[features],ScaleTemplates,likelihoodmethod)
            part['LikeliestScale'] = LikeliestScale[0][0]

def Evaluate_ScaleLikelihoodEstimation(dataDictionary, scaleTypes):
    SCALES = []
    SCALES_PREDICT = []
    for files, parts in dataDictionary.items():
        for part in parts:
            SCALES.append(part['groundtruth']['scaleType'].split(':')[1])
            SCALES_PREDICT.append(part['LikeliestScale'])
    f1_score, accuracy_score = EvaluatePredictions(SCALES, SCALES_PREDICT)
    
    print('F measure (weighted) :  \n' + str(f1_score*100) + ' %')
    print('Overall Accuracy : \n' + str(accuracy_score*100) + ' %' )
    CONFUSIONMATRIX = confusion_matrix(SCALES,SCALES_PREDICT,scaleTypes)
    plot_confusion_matrix(CONFUSIONMATRIX,scaleTypes)   
    
############ CHORD-SCALE DETECTION - METHOD 2: MACHINE LEARNING ####################

def Classification_SVM(filename,dataDir):
      
            
    numBins = 12 
    
    list_accuracy=[]
    
    df = pd.read_csv(os.path.join(dataDir,filename))
    df.pop('name'); dataclass=df.pop('scaleType')
    X=df; Y=dataclass
    modeSet = set(Y)
    
    cm,acc,f = machineLearningEvaluation(dataDir,X,Y,numBins)

    modeSet = sorted(modeSet)
    print('\n')
    
    return acc, f, cm, modeSet

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

        f_measure, accscore = EvaluatePredictions(Y_test, Y_pred)

        f_measures.append(f_measure)
        accuracies.append(accscore)
        for i in Y_test:
            Y_total.append(i)
        for i in Y_pred:
            Y_pred_total.append(i)

    print('Accuracy score for the Feature Set : \n')
    scaleTypes = ['major','dorian','phrygian','lydian','mixolydian','minor','locrian','melmin','lydianb7','altered','hminor', 'hwdiminished']
    ##AVERAGE ALL RANDOM SEED ITERATIONS FOR GENERALIZATION
    print('F-measure (mean,std) --- FINAL \n')
    f = round(np.mean(f_measures) ,2)
    fstd = np.std(f_measures)
    print(f,fstd )
    print('Accuracy (mean,std) FINAL \n')
    ac = round(np.mean(accuracies), 2)
    accstd=np.std(accuracies)
    print(ac,accstd)
    cm = confusion_matrix(Y_total, Y_pred_total,scaleTypes)


    with open(targetDir + 'scores_' + str(numBin) + '.txt', 'w') as scorefile:
        scorefile.write(str(f))
        scorefile.write(str(ac))
        
    return cm, f_measures, accuracies

def EvaluatePredictions(Y_test, Y_pred):
    
    return (f1_score(Y_test, Y_pred, average='weighted'), accuracy_score(Y_test, Y_pred))

############ VISUALIZATION - CLASSIFICATION ################

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
    plt.figure(figsize=(7, 8))
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
    
def plot_VIOLINPLOT(DATA_ACCURACY, DATA_FSCORE):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    pos = [1,2,3]
    
    x_ticks = ['HPCP(mean)', 'HPCP(std)', 'HPCP(mean+std)']
    
    axes[0].violinplot(DATA_ACCURACY, points=80, widths=0.7,
                       showextrema=True, showmedians=True)
    axes[0].set_title('ACCURACY SCORE', fontsize = 18)
    axes[0].set_xlabel('FEATURE SET', fontsize = 18)
    axes[0].get_xaxis().set_tick_params(direction='out')
    axes[0].xaxis.set_ticks_position('bottom')
    axes[0].set_xticks(np.arange(1, len(x_ticks) + 1))
    axes[0].set_xticklabels(x_ticks,rotation=45, fontsize = 14)
    axes[0].set_xlim(0.25, len(x_ticks) + 0.75)
    
    parts = axes[1].violinplot(DATA_ACCURACY, points=80, widths=0.7,
                       showextrema=True, showmedians=True)
    axes[1].set_title('F SCORE', fontsize = 18)
    axes[1].set_xlabel('FEATURE SET', fontsize = 18)
    axes[1].get_xaxis().set_tick_params(direction='out')
    axes[1].xaxis.set_ticks_position('bottom')
    axes[1].set_xticks(np.arange(1, len(x_ticks) + 1))
    axes[1].set_xticklabels(x_ticks,rotation=45, fontsize = 14)
    axes[1].set_xlim(0.25, len(x_ticks) + 0.75)


    plt.show()    

#######  SINGLE FILE PROCESSING ###########

def FeatureExtraction_single(fileName, params,annotationFile):
    '''
    fileName : input audio file (.mp3)
    fileDir : directory of the audio file
    params : Analysis parameters object
    annotationData : Data from the annotation (exercise.json)
    '''
    
    numBins = params.numBins    
    fs=params.fs
    
    x = ess.MonoLoader(filename = fileName, sampleRate = fs)()
    x = ess.DCRemoval()(x) ##preprocessing / apply DC removal for noisy regions
    x = ess.EqualLoudness()(x)
    #Windowing (first converting from msec to number of samples)
    windowSize=round(fs*params.windowSize/1000);windowSize=int(windowSize/2)*2#assuring window size is even
    hopSize=round(fs*params.hopSize/1000);hopSize=int(hopSize/2)*2#assuring hopSize is even
    
    with open(annotationFile) as json_file:
        annotationData=json.load(json_file)
    partsList = dict()  
    
    for part in annotationData['parts']:
        fileData=initiateData4File(part['name'],fileName.split('/')[0])
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
            tunefreq, tunecents = ess.TuningFrequency()(freq,mag) ## estimate tuning fo HPCP vectors
            reffreq = computeReferenceFrequency(fileData['key'],tunefreq) ## compute Reference frequency of the reference bin in HPCP vectors
            HPCPvector = ess.HPCP(normalized='unitSum',referenceFrequency=reffreq,size = numBins, windowSize = 12/numBins)(freq,mag) #harmonic pitch-class profiles 
            
            HPCPmaxOnly = np.zeros_like(HPCPvector) ### Keep ONLY the HPCP bin with max value, set other to 0 (zero).
            HPCPmaxOnly[np.argmax(HPCPvector)] = np.max(HPCPvector)
            hpcp.append(HPCPmaxOnly)
        
        ### TRANSIENT REMOVAL  --- IF the non-zero bin of HPCP vectors of current, previous and next do not correspond 
        ### to the same position in the HPCP vectors, set the current HPCP bin to zero. 
        ### (basically removes short durations that could be caused by transients, noises or artifacts)
        
        for i in range(1,len(hpcp)-1):
            if np.argmax(hpcp[i]) != np.argmax(hpcp[i-1]) and np.argmax(hpcp[i]) != np.argmax(hpcp[i+1]):
                hpcp[i] = np.zeros(12) 
            
        fileData['hpcp']=np.array(hpcp)    
        
        for j in range(numBins):
            hpcps = [];
            for i in range(len(hpcp)):
                hpcps.append(hpcp[i][j])
            fileData['mean_hpcp_vector'].append(np.mean(hpcps))
            fileData['std_hpcp_vector'].append(np.std(hpcps))
            
        partsList[part['name']] = fileData

                
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
    
####### CASE STUDY : CHORD-SCALE EXERCISE ###############

def SegmentExerciseAudio(FILENAME, startTime, endTime , params):
    
    startTime = float(startTime); endTime = float(endTime)
    audio = ess.MonoLoader(filename = FILENAME, sampleRate = params.fs)()

    audio_PART = audio[params.fs*startTime:params.fs*endTime]
    
    return(audio_PART)

######## PERFORMANCE ASSESSMENT & GRADING ##############

def ComputeCosineSimilarity(v1,v2):
    '''
    Compute Cosine Similarity of v1 to v2 using following equation:
    (v1 dot v2)/{||v1||*||v2||)
    
        This metric is used to measure the angular distance between the templates of the 'GROUND_TRUTH' and 'STUDENT_PERFORMANCE' scales.
    
    '''
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return (sumxy/math.sqrt(sumxx*sumyy))

def ComputeInScaleRate(ChromaVector, ScaleArray):
    
    return np.sum(np.multiply(ChromaVector,ScaleArray)/np.sum(ChromaVector))

def ComputeScaleCompleteness(ChromaVector, ScaleArray):
               
    return (np.count_nonzero(np.multiply(ChromaVector,ScaleArray)))/np.count_nonzero(ScaleArray)

def PerformanceAssessment(StudentData, likeliestScale, ScaleTemplates):
    
    ExpectedScale = StudentData['scaleType']
    scaleArrayExpected = ScaleTemplates[ExpectedScale]['scaleArray']
    scaleArrayStudent = ScaleTemplates[likeliestScale]['scaleArray']
    chromaVector = StudentData['mean_hpcp_vector']
    stdchromaVector = StudentData['std_hpcp_vector']
    
    inScaleRate = ComputeInScaleRate(chromaVector,scaleArrayExpected)
    
    scaleCompleteness = ComputeScaleCompleteness(chromaVector, scaleArrayExpected)
    
    NONZERO_PITCHES = np.count_nonzero(chromaVector)
    
    if NONZERO_PITCHES < np.sum(scaleArrayExpected)-2 :                
        
        return(inScaleRate,'N/A' , scaleCompleteness)
    
    elif NONZERO_PITCHES < np.sum(scaleArrayExpected) and inScaleRate > 90 :
        
        return(inScaleRate, '100', scaleCompleteness)
    
    elif NONZERO_PITCHES < np.sum(scaleArrayExpected) and inScaleRate < 90 :
        scalechoicecorrectness = ComputeCosineSimilarity(scaleArrayExpected,scaleArrayStudent)
        return(inScaleRate, scalechoicecorrectness, scaleCompleteness)
    
    else:    
        scalechoicecorrectness = ComputeCosineSimilarity(scaleArrayExpected,scaleArrayStudent)    
        return inScaleRate, scalechoicecorrectness, scaleCompleteness

############# ANALYSIS ON SEPERATED REGIONS  #############

def SegmentAnalysis(ExercisePart, ScaleTemplates):
    
    EstimationMethod = 2 #additive likelihood
    
    likelihoodsParts = []
    EstimatedScales = []

    PART_HPCP = ExercisePart['hpcp']
    hpcpVector = np.zeros_like(PART_HPCP)
    likelihoodsVector = []

    for k in range(len(PART_HPCP)):
        hpcpVector = hpcpVector + PART_HPCP[k]
        LikeliestScale, LikelihoodsArray = ScaleLikelihoodEstimation(hpcpVector, ScaleTemplates, EstimationMethod)
        framelikelihoods = []
        for j in range(len(LikelihoodsArray)):
            framelikelihoods.append(LikelihoodsArray[j][1]['likelihood'])
        framelikelihoods = np.array(framelikelihoods).reshape(1,-1)
        framelikelihoods = framelikelihoods[0]
        likelihoodsVector.append(framelikelihoods)
        
    scalesList = ['major','dorian','phrygian','lydian','mixolydian','minor','locrian','melmin','lydianb7','altered','hminor','wholetone','hwdiminished']
    
    '''
    ### Plotting and scale estimations on cumulated likelihood vectors
    print('The most likelihood scale of the student performance in ' + ExercisePart['name'] + ' is : \n')
    print(LikeliestScale[0],'\n')
   
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(8,5))
    
    ax1.imshow(np.transpose(likelihoodsVector),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma',norm=plt.Normalize())
    ax1.set_title(ExercisePart['key'] + '-' + ExercisePart['scaleType'] + ' Scale',fontsize = 16)
    ax1.set_xlabel('Frame #')
    ax1.set_ylabel('ScaleTypes')
    tick_marks = np.arange(len(scalesList))
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(scalesList)
    
    pitch_classes = ['A','Bb','B','C','C#','D','D#','E','F','F#','G','G#']
           
    ax2.imshow(np.transpose(PART_HPCP),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma')
    ax2.set_xlabel('Frame #')
    ax2.set_ylabel('Pitch-Classes')
    tick_marks1 = np.arange(len(pitch_classes))
    ax2.set_yticks(tick_marks1)
    
    plt.show()
    '''
    return LikeliestScale[0], likelihoodsVector
                    

def ScaleEstimationCumulative(FeatureData,ScaleTemplates,EstimationMethod):
    
    likelihoodsParts = []
    EstimatedScales = []
    
    for i in range(len(FeatureData)):  
        PartDataHPCP = FeatureData['Part'+str(i+1)]['hpcp']
        hpcpAgg = np.zeros_like(PartDataHPCP)
        likelihoodsVector = []
        for k in range(len(PartDataHPCP)):
            hpcpAgg = hpcpAgg + PartDataHPCP[k]
            maxscalelike, likelihood = ScaleLikelihoodEstimation(hpcpAgg, ScaleTemplates, EstimationMethod)
            framelikelihoods = []
            for j in range(len(likelihood)):
                framelikelihoods.append(likelihood[j][1]['likelihood'])

            framelikelihoods = np.array(framelikelihoods).reshape(1,-1)
            framelikelihoods = framelikelihoods[0]
            likelihoodsVector.append(framelikelihoods)    
            
        EstimatedScales.append(maxscalelike)    
        likelihoodsParts.append(likelihoodsVector)
    
    length =int(len(likelihoodsParts)/2)

    scaletypes = []
    for item in ScaleTemplates:
        scaletypes.append(item)
        scaleTypes = sorted(scaletypes)
        
        
    ### Plotting and scale estimations on cumulated likelihood vectors
    
    for i in range(length):
        
        print('The most likelihood scale of the student performance in Part' + str(2*i+1) + ' is : \n')
        print(EstimatedScales[2*i][0][0],'\n')
              
        print('The most likelihood scale of the student performance in Part' + str(2*i+2) + ' is : \n')      
        print(EstimatedScales[2*i+1][0][0],'\n')
              
        likelihoodConcat = np.concatenate((likelihoodsParts[2*i],likelihoodsParts[2*i+1]),axis = 0)
        fig = plt.figure(figsize=(7,4))
        figure = plt.imshow(np.transpose(likelihoodConcat),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma',norm=plt.Normalize())
        plt.xlabel('Frame #')
        plt.ylabel('ScaleType')
        tick_marks = np.arange(len(scaleTypes))
        plt.yticks(tick_marks, scaleTypes)
        cbar = fig.colorbar(figure,  ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['0', '0.5', '1'])
        plt.show()
        
    ### NORMALIZE THE FINAL SCALE-LIKELIHOOD VECTOR W.R.T 'UNIT_SUM' NORM
    
    #for i in range(len(framelikelihoods)):
        #framelikelihoods[i] = framelikelihoods[i]/np.sum(framelikelihoods)
        
    return framelikelihoods
    
######### TEMPORAL ANALYSIS ##############
    
def ScaleEstimationAggregate(FeatureData, winSize, hopSize, ScaleTemplates):
    
    likelihoodsParts = []
    for i in range(len(FeatureData)):
        likelihoods = []
        hpcpVec = np.concatenate((FeatureData['Part'+str(i+1)]['hpcp'],np.zeros((winSize-hopSize,12))),axis = 0)
        #print(hpcpVec)
        for k in range(len(FeatureData['Part'+str(i+1)]['hpcp'])):
            hpcpAgg = np.zeros_like(hpcpVec[0])
            for l in range(winSize-hopSize):
                hpcpAgg = hpcpAgg + hpcpVec[k+l]
                #print(hpcpAgg)
            maxscalelike, likelihood = ScaleLikelihoodEstimation(hpcpAgg, ScaleTemplates, 2)
            framelikelihoods = []
            for j in range(len(likelihood)):
                framelikelihoods.append(likelihood[j][1]['likelihood'])
            framelikelihoods = normalize(np.array(framelikelihoods).reshape(1,-1),norm = 'l2')
            framelikelihoods = framelikelihoods[0]
            likelihoods.append(framelikelihoods)    
        likelihoodsParts.append(likelihoods)
        
    length =int(len(likelihoodsParts)/2)
    
    scaletypes = []
    for item in ScaleTemplates:
        scaletypes.append(item)
    scaleTypes = sorted(scaletypes)    
    for i in range(length):
        likelihoodConcat = np.concatenate((likelihoodsParts[2*i][:(winSize-hopSize)],likelihoodsParts[2*i+1][:(winSize-hopSize)]),axis = 0)
        
        fig = plt.figure(figsize=(7,4))
        figure = plt.imshow(np.transpose(likelihoodConcat),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma',norm=plt.Normalize())
        plt.xlabel('Frame #')
        plt.ylabel('ScaleType')
        tick_marks = np.arange(len(scaleTypes))
        plt.yticks(tick_marks, scaleTypes)
        cbar = fig.colorbar(figure,  ticks=[0, 0.5, 1])
        cbar.ax.set_yticklabels(['0', '0.5', '1'])
        plt.show()
     
    ### NORMALIZE THE FINAL SCALE-LIKELIHOOD VECTOR W.R.T 'UNIT_SUM' NORM
    
    #for i in range(len(framelikelihoods)):
     #   framelikelihoods[i] = framelikelihoods[i]/np.sum(framelikelihoods)
        
    return framelikelihoods
        
#########################################

def SegmentAnalysis1(ExercisePart, ScaleTemplates):
    
    EstimationMethod = 2 #additive likelihood
    
    likelihoodsParts = []
    EstimatedScales = []
    
    PART_HPCP = ExercisePart['hpcp']
    hpcpVector = np.zeros_like(PART_HPCP)
    likelihoodsVector = []

    scaletypes = []
    for item in ScaleTemplates:
        scaletypes.append(item)
        scaleTypes = sorted(scaletypes)

    for k in range(len(PART_HPCP)):
        hpcpVector = hpcpVector + PART_HPCP[k]
        LikeliestScale, LikelihoodsArray = ScaleLikelihoodEstimation(hpcpVector, ScaleTemplates, EstimationMethod)
        framelikelihoods = []
        for j in range(len(LikelihoodsArray)):
            framelikelihoods.append(LikelihoodsArray[j][1]['likelihood'])
        framelikelihoods = np.array(framelikelihoods).reshape(1,-1)
        framelikelihoods = framelikelihoods[0]
        likelihoodsVector.append(framelikelihoods)        
    
    return LikeliestScale[0]

def exerciseAssessment(audioFile, annotationFile,SELECTION_PARAMETER):
    '''
    INPUT:
    
    audioFile : directory to the audio file of analysis
    
    annotationFile : directory to the annotation file of the exercise (scalesexercise.json)
    
    SELECTION_PARAMETER : PARAMETER FOR OUTPUT FORMAT SELECTION. IF '0', returns OVERALL_GRADES, IF '1', returns table of separate grades of each region. 
    
    OUTPUT:
    
    '''
    
    params = AnalysisParams(200,100,'hann',2048,44100,12)
    
    ScaleTemplates = ScaleDictionary()
    
    FEATURES_DATA = FeatureExtraction_single(audioFile, params, annotationFile)
    GRADES = []
    GRADES.append(['PARTS', ('In-Scale_Rate', 'Scale_Correctness','In-Scale_Completeness')])
    
    for i in range(len(FEATURES_DATA)):
        PartData = FEATURES_DATA['Part'+str(i+1)]
        
        LikeliestScale = SegmentAnalysis1(PartData,ScaleTemplates)
        
        GRADES.append(['Part'+str(i+1),PerformanceAssessment(PartData,LikeliestScale,ScaleTemplates)])
      
    INSCALERATES = [] ; INSCALECOMPLETENESS = [] ; SCALECORRECTNESS = []
    for l in range(1,len(GRADES)):
        
        INSCALERATES.append(GRADES[l][1][0])
        
        if GRADES[l][1][1] == 'N/A':
            INSCALECOMPLETENESS.append(0)
            
        else:
            INSCALECOMPLETENESS.append(GRADES[l][1][1])
        SCALECORRECTNESS.append(GRADES[l][1][2])
    
    if SELECTION_PARAMETER == 0 :        
        return(np.mean(INSCALERATES) ,round(np.mean(INSCALECOMPLETENESS),3)*100,np.mean(SCALECORRECTNESS))
    
    elif SELECTION_PARAMETER == 1 :    
        return(GRADES)    


def VisualizePerformance(LikelihoodsVectors, FeaturesData):
    
    scalesList = ['major','dorian','phrygian','lydian','mixolydian','minor','locrian','melmin','lydianb7','altered','hminor','wholetone','hwdiminished']
    
    fig, axes = plt.subplots(3, 4, sharey=True,figsize=(12,8))
    countPART = 0
    for row in range(3):
        column = 0    
        PART_DATA = FeaturesData['Part'+str(countPART+1)]
        axes[row,column].imshow(np.transpose(LikelihoodsVectors[countPART]),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma',norm=plt.Normalize())
        axes[row,column].set_title('Part'+str(countPART+1) + ' - Target: ' + PART_DATA['key'] + '-' + PART_DATA['scaleType'] ,fontsize = 14)
        axes[row,column].set_xlabel('Frame #')
        axes[row,column].set_ylabel('ScaleTypes')
        tick_marks = np.arange(len(scalesList))
        axes[row,column].set_yticks(tick_marks)
        axes[row,column].set_yticklabels(scalesList)

        pitch_classes = ['A','Bb','B','C','C#','D','D#','E','F','F#','G','G#']

        axes[row,column+1].imshow(np.transpose(FeaturesData['Part'+str(countPART+1)]['hpcp']),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma')
        axes[row,column+1].set_xlabel('Frame #')
        axes[row,column+1].set_ylabel('Pitch-Classes')
        axes[row,column+1].set_title('Chroma Features vs Time')
        tick_marks1 =  np.arange(len(pitch_classes))
        
        axes[row,column+1].set_yticks(tick_marks1)
        
        countPART = countPART + 1 
        PART_DATA = FeaturesData['Part'+str(countPART+1)]
        axes[row,column+2].imshow(np.transpose(LikelihoodsVectors[countPART]),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma',norm=plt.Normalize())
        axes[row,column+2].set_title('Part'+str(countPART+1) + ' - Target: ' + PART_DATA['key'] + '-' + PART_DATA['scaleType'] + ' Scale',fontsize = 14)
        axes[row,column+2].set_xlabel('Frame #')
        axes[row,column+2].set_ylabel('ScaleTypes')
        tick_marks = np.arange(len(scalesList))
        axes[row,column+2].set_yticks(tick_marks)
        axes[row,column+2].set_yticklabels(scalesList)
        
        axes[row,column+3].imshow(np.transpose(FeaturesData['Part'+str(countPART+1)]['hpcp']),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma')
        axes[row,column+3].set_xlabel('Frame #')
        axes[row,column+3].set_ylabel('Pitch-Classes')
        axes[row,column+3].set_title('Chroma Features vs Time')
        tick_marks1 = np.arange(len(pitch_classes))
        axes[row,column+3].set_yticks(tick_marks1)
        
        countPART = countPART + 1
        
    plt.tight_layout()
    plt.show()
    
def plot_tableGRADES(GRADES, ESTIMATEDSCALES):
    
    parts = []; inscalerates = []; inscalecompleteness = [] ; scalecorrectness = [];
    ESTIMATEDSCALES.append(' ----- ')
    fig, ax = plt.subplots()

    ax.axis('off')
    ax.axis('tight')

    for i in range(1,len(GRADES)):
        parts.append(GRADES[i][0])
        inscalerates.append(round(GRADES[i][1][0],3)*100)
        if type(GRADES[i][1][1]) == float:
            inscalecompleteness.append(GRADES[i][1][1]*100)
        else:
            inscalecompleteness.append(GRADES[i][1][1])
        scalecorrectness.append(round(GRADES[i][1][2]*100,3))

    TABLE = plt.table(cellText=np.transpose([inscalerates,inscalecompleteness,scalecorrectness,ESTIMATEDSCALES]),
                          rowLabels= parts,
                          colLabels= ['In-Scale_Rate (%) ', 'Scale_Correctness (%) ','In-Scale_Completeness (%) ','Estimated_Scales (%) '], loc='center right')
    
    TABLE.set_fontsize(20)
    TABLE.scale(2, 2)
    #plt.subplots_adjust(left=0.6, right = 0.7, wspace=0.2, top=0.65, bottom = 0.6)
    fig.tight_layout()
    plt.show()    