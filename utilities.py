import os, sys, csv, pickle, itertools, warnings, json, math
import numpy as np
import essentia.standard as ess
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import IPython.display as ipd
import pandas as pd
from pandas import DataFrame, read_csv

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score
from sklearn.preprocessing import normalize


######## ARGUMENT PARSING ########


####### DOWNLOAD DATASET #########


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
    
    for j in range(len(fileData['hpcp'][0])):
        hpcps = [];
        for i in range(len(fileData['hpcp'])):
            hpcps.append(fileData['hpcp'][i][j])
        fileData['mean_hpcp_vector'].append(np.mean(hpcps))
        
        fileData['std_hpcp_vector'].append(np.std(hpcps))   
    
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
        
#############




        
############ CHORD - SCALE DETECTION - METHOD 1: TEMPLATE-BASED LIKELIHOOD ESTIMATION ##############


def ScaleLikelihoodEstimation(ChromaVector, ScaleTemplates, method):
    
    for scale in ScaleTemplates.items():
        #scale[0] : scale name (scaleTemplates.keys())
        #scale[1] : elements in scale dictionaries
        
        if method == 1:
        
            scale[1]['likelihood'] = Maxlikelihood_MULTIPLICATION(ChromaVector,scale[1]['scaleArray'])
        
        elif method == 2:
    
            scale[1]['likelihood'] = Maxlikelihood_SUMMATION(ChromaVector,scale[1]['scaleArray'])
    
    maxLikelihood = ['NA',0]
    likelihoods = []
    for item in ScaleTemplates.items():
        if item[1]['likelihood']>maxLikelihood[1]:
            maxLikelihood[0]=item;maxLikelihood[1] = item[1]['likelihood']
        likelihoods.append(item)
        sortedlikelihoods = sorted(likelihoods, key = lambda k:k)
    return(maxLikelihood,sortedlikelihoods)

def Maxlikelihood_MULTIPLICATION(ChromaVector,scaleArray):
    
    NumNotesScale = np.sum(scaleArray)
    ChromaScaled = np.power(ChromaVector,scaleArray)    
    scale_likelihood = np.prod(ChromaScaled) / ((1/NumNotesScale)**NumNotesScale)

    return(scale_likelihood) 
    
def Maxlikelihood_SUMMATION(ChromaVector,scaleArray):
    
    NumNotesScale = np.sum(scaleArray)
    ChromaScaled = np.multiply(ChromaVector,scaleArray)        
    scale_likelihood = np.sum(ChromaScaled) / NumNotesScale        

    return(scale_likelihood)

############ VISUALIZATION - SCALE LIKELIHOODS ########################

def VisualizeChromaANDScaleLikelihoods(filename, soundname, dataIndex, scaleTemplates, likelihoodmethod):
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    data1 = data[soundname]
    hpcps = data1[dataIndex]['hpcp']
    hpcpAgg = np.zeros_like(hpcps[0])

    scalelikelihoods = []
    scaletypes = []
    for i in range(len(hpcps)):
        hpcpAgg = hpcpAgg + hpcps[i]
        #print(hpcpAgg)
        maxscalelike, likelihood = ScaleLikelihoodEstimation(hpcpAgg,scaleTemplates, likelihoodmethod)        

        framelikelihoods = []
        for j in range(len(likelihood)):
            framelikelihoods.append(likelihood[j][1]['likelihood'])
            scaletypes.append(likelihood[j][0])          
        scalelikelihoods.append(framelikelihoods)
        
    scaletypes = set(scaletypes)  
    scaleTypes = sorted(scaletypes)
    
    pitch_classes = ['A','Bb','B','C','C#','D','D#','E','F','F#','G','G#']
    
    
    MaxLikelihoodNormalized = maxscalelike[1] / np.sum(scalelikelihoods[-1])
    print('Maximum Likeliest Scale of Phrase : ' + str(maxscalelike[0][0]) + '    with likeliest : ' + str(MaxLikelihoodNormalized))
    #print(scalelikelihoods)
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
    
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
    
    plt.show()
        
############ CHORD-SCALE DETECTION - METHOD 2: MACHINE LEARNING ####################

def Classification_SVM(filename,dataDir):
      
        
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
            HPCPvector = ess.HPCP(normalized='unitSum',referenceFrequency=reffreq,size = numBins, windowSize = 12/numBins)(freq,mag) #harmonic pitch-class profiles 
            HPCPmaxOnly = np.zeros_like(HPCPvector)
            HPCPmaxOnly[np.argmax(HPCPvector)] = np.max(HPCPvector)
            hpcp.append(HPCPmaxOnly)
         
        fileData['hpcp']=np.array(hpcp)
        
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
    #for i in range(numBins):
     #   ind=str(i)
      #  fieldnames.append('std_hpcp'+ind)    
    fieldnames.append('scaleType')
    
    dataList=[]
    dataList.append(fieldnames)
    
    for part in partsList.items(): ##navigate in dictionary
         
        tempList=[] #temporary List to put attributes for each audio slice (data-point)
        
        dataname = part[0] #name of data
        tempList.append(dataname)
        for i in range(numBins): #append mean_HPCP vector bins separately               
            tempList.append(part[1]['mean_hpcp_vector'][i])
        #for i in range(numBins): #append mean_HPCP vector bins separately            
         #   tempList.append(part[1]['std_hpcp_vector'][i])    
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
    
####### CASE STUDY : CHORD-SCALE EXERCISE ###############

def SegmentExerciseAudio(FILENAME, Features_Part, params):
    
    startTime = float(Features_Part['startTime']); endTime = float(Features_Part['endTime'])
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
    return sumxy/math.sqrt(sumxx*sumyy)

def ComputeInScaleRate(ChromaVector, ScaleArray):
    
    return np.sum(np.multiply(ChromaVector,ScaleArray)/np.sum(ChromaVector))

def ComputeScaleCompleteness(ChromaVector, ScaleArray):
    
    return (np.count_nonzero(np.multiply(ChromaVector,ScaleArray)))/(np.count_nonzero(ScaleArray))

def PerformanceAssessment(StudentData, likeliestScale, ScaleTemplates):
    
    ExpectedScale = StudentData['scaleType']
    scaleArrayExpected = ScaleTemplates[ExpectedScale]['scaleArray']
    scaleArrayStudent = ScaleTemplates[likeliestScale]['scaleArray']
    chromaVector = StudentData['mean_hpcp_vector']
    stdchromaVector = StudentData['std_hpcp_vector']
    
    inScaleRate = ComputeInScaleRate(chromaVector,scaleArrayExpected)
    
    scalechoicecorrectness = ComputeCosineSimilarity(scaleArrayExpected,scaleArrayStudent)
    
    scaleCompleteness = ComputeScaleCompleteness(stdchromaVector, scaleArrayExpected)
    
    return inScaleRate, scalechoicecorrectness, scaleCompleteness

############# ANALYSIS ON SEPERATED REGIONS  #############

def SegmentAnalysis(ExercisePart, ScaleTemplates):
    
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

    ### Plotting and scale estimations on cumulated likelihood vectors
    print('The most likelihood scale of the student performance in ' + ExercisePart['name'] + ' is : \n')
    print(LikeliestScale[0][0],'\n')
   
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(8,5))
    
    ax1.imshow(np.transpose(likelihoodsVector),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma',norm=plt.Normalize())
    ax1.set_title(ExercisePart['key'] + '-' + ExercisePart['scaleType'] + ' Scale',fontsize = 16)
    ax1.set_xlabel('Frame #')
    ax1.set_ylabel('ScaleTypes')
    tick_marks = np.arange(len(scaleTypes))
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(scaleTypes)
    
    pitch_classes = ['A','Bb','B','C','C#','D','D#','E','F','F#','G','G#']
           
    ax2.imshow(np.transpose(PART_HPCP),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma')
    ax2.set_xlabel('Frame #')
    ax2.set_ylabel('Pitch-Classes')
    tick_marks1 = np.arange(len(pitch_classes))
    ax2.set_yticks(tick_marks1)
    
    plt.show()
    
    return LikeliestScale[0][0]
                    

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
        
        