import os, sys, pickle, csv, json, itertools, vamp, math
import numpy as np
import essentia
import essentia.standard as ess
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import IPython.display as ipd

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
    
    fileData['NNLS'] = [];
    fileData['mean_NNLS']=[];
    fileData['std_NNLS']=[];
    
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
                        

######### FEATURE EXTRACTION - HARMONIC PITCH CLASS PROFILES #################


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
        HPCPvector = ess.HPCP(normalized='unitSum',referenceFrequency=reffreq,size = numBins, windowSize = 12/numBins)(freq,mag) #harmonic pitch-class profiles 
        HPCPmaxOnly = np.zeros_like(HPCPvector)
        HPCPmaxOnly[np.argmax(HPCPvector)] = np.max(HPCPvector)
        hpcp.append(HPCPmaxOnly)
        
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
        
######### FEATURE EXTRACTION - NNLS CHROMA #################        
      
def computeFeaturesNNLS(audiofile, fileData,params, sampleRate=44100, stepSize=2048):
    mywindow = np.array(
        [0.001769, 0.015848, 0.043608, 0.084265, 0.136670, 0.199341, 0.270509, 0.348162, 0.430105, 0.514023,
         0.597545, 0.678311, 0.754038, 0.822586, 0.882019, 0.930656, 0.967124, 0.990393, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999650, 0.996856, 0.991283,
         0.982963, 0.971942, 0.958281, 0.942058, 0.923362, 0.902299, 0.878986, 0.853553, 0.826144,
         0.796910, 0.766016, 0.733634, 0.699946, 0.665140, 0.629410, 0.592956, 0.555982, 0.518696,
         0.481304, 0.444018, 0.407044, 0.370590, 0.334860, 0.300054, 0.266366, 0.233984, 0.203090,
         0.173856, 0.146447, 0.121014, 0.097701, 0.076638, 0.057942, 0.041719, 0.028058, 0.017037,
         0.008717, 0.003144, 0.000350])
    
    
    audio = ess.MonoLoader(filename=audiofile, sampleRate=sampleRate)()
    audio = ess.DCRemoval()(audio) ##preprocessing / apply DC removal for noisy regions
    audio = ess.EqualLoudness()(audio)
    
    startTime=fileData['groundtruth']['startTime']
    endTime=fileData['groundtruth']['endTime']
    audio_slice = sliceAudiotoParts(audio,endTime,startTime,params)
    
    '''
    stepsize, semitones = vamp.collect(
        audio_slice, sampleRate, "nnls-chroma:nnls-chroma", parameters = {'chromanormalize':2} , output="semitonespectrum", step_size=stepSize)["matrix"]
    '''
    
    stepsize, semitones = vamp.collect(
        audio_slice, sampleRate, "nnls-chroma:nnls-chroma" , output="semitonespectrum", step_size=stepSize)["matrix"]
    
    chroma = np.zeros((semitones.shape[0], 12))
    for i in range(semitones.shape[0]):
        tones = semitones[i] * mywindow
        cc = chroma[i]
        for j in range(tones.size):
            cc[j % 12] = cc[j % 12] + tones[j]
            
    keys = {'A':0,'Bb':1,'B':2,'C':3,'C#':4,'D':5,'Eb':6,'E':7,'F':8,'F#':9,'G':10,'Ab':11}
    for key in keys:
        if key == fileData['groundtruth']['scaleType'].split(':')[0]:
            pitch_shift = keys[key]
    pitch_shift = pitch_shift * -1                
    # roll from 'A' based to 'C' based
    chroma = np.roll(chroma, shift=pitch_shift, axis=1)
    '''
    for i in range(len(chroma)):
        for j in range(params.numBins):            
            chroma[i][j] = chroma[i][j]/np.sum(chroma[i])
    '''
    fileData['NNLS'] = chroma
    

def computeFeaturesNNLSGlobal(fileData,params):
    
    computeFeaturesNNLS(os.path.join('audio/', fileData['name']+'.mp3'), fileData,params)
    chromaMean = []
    chromaSTD = []
    for j in range(fileData['numBins']):
        chromas = [];
        for i in range(len(fileData['NNLS'])):
            chromas.append(fileData['NNLS'][i][j])
            
        chromaMean.append(np.mean(chromas))
        chromaSTD.append(np.std(chromas))
        
    for j in range(fileData['numBins']):
        chromaMean[j]=chromaMean[j]/np.sum(chromaMean)
        chromaSTD[j]=chromaSTD[j]/np.sum(chromaSTD)

    fileData['mean_NNLS'] = chromaMean
    fileData['std_NNLS'] = chromaSTD
    
                
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
        
def generateCSVNNLS(filename, dataDir):
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    #numBins = data['toprak_dorian'][0]['numBins']    
    numBins = 12
    fieldnames=['name']
    for i in range(numBins):
        ind=str(i)
        fieldnames.append('mean_NNLS'+ind)
    for i in range(numBins):
        ind=str(i)
        fieldnames.append('std_NNLS'+ind)    
    fieldnames.append('scaleType')
    dataList=[]
    dataList.append(fieldnames)
    for fileName, parts in data.items(): ##navigate in dictionary
        for part in parts: #search within audio slices
            tempList=[] #temporary List to put attributes for each audio slice (data-point)
            dataname = part['name']+'_'+part['groundtruth']['name'] #name of data
            tempList.append(dataname)
            for i in range(numBins): #append mean_HPCP vector bins separately            
                tempList.append(part['mean_NNLS'][i])
            for i in range(numBins): #append mean_HPCP vector bins separately            
                tempList.append(part['std_NNLS'][i])    
            tempList.append(part['groundtruth']['scaleType'].split(':')[1])    #append scales for classification
            dataList.append(tempList)

    with open(dataDir+'CSVfilefor_NNLS.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataList)            
        
############## MAXIMUM LIKELIHOOD #################

def maxlikelihood(ChromaVector):
    
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
    ScaleTemplates['mminor'] = {'scaleArray':[1,0,1,1,0,1,0,1,0,1,0,1]}
    ScaleTemplates['hminor'] = {'scaleArray':[1,0,1,1,0,1,0,1,1,0,0,1]}
    ScaleTemplates['hwdiminished'] = {'scaleArray':[1,1,0,1,1,0,1,1,0,1,1,0]}
    
    
    for scale in ScaleTemplates.items():
        #scale[0] : scale name (scaleTemplates.keys())
        #scale[1] : elements in scale dictionaries
        
        NumNotesScale = np.sum(scale[1]['scaleArray'])
        #print(NumNotesScale)
        ChromaScaled = np.power(ChromaVector,scale[1]['scaleArray'])
        
        scale[1]['likelihood'] = np.prod(ChromaScaled) / ((1/NumNotesScale)**NumNotesScale)
        
        #print(scale[1]['likelihood'])
    
    maxLikelihood = ['NA',0]
    likelihoods = []
    for item in ScaleTemplates.items():
        if item[1]['likelihood']>maxLikelihood[1]:
            maxLikelihood[0]=item;maxLikelihood[1] = item[1]['likelihood']
        likelihoods.append(item)
        sortedlikelihoods = sorted(likelihoods, key = lambda k:k)
    return(maxLikelihood,sortedlikelihoods) 
    
def maxlikelihood2(ChromaVector):
    
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
    ScaleTemplates['mminor'] = {'scaleArray':[1,0,1,1,0,1,0,1,0,1,0,1]}
    ScaleTemplates['hminor'] = {'scaleArray':[1,0,1,1,0,1,0,1,1,0,0,1]}
    ScaleTemplates['hwdiminished'] = {'scaleArray':[1,1,0,1,1,0,1,1,0,1,1,0]}
    
    
    for scale in ScaleTemplates.items():
        #scale[0] : scale name (scaleTemplates.keys())
        #scale[1] : elements in scale dictionaries
        
        NumNotesScale = np.sum(scale[1]['scaleArray'])
        #print(NumNotesScale)
        ChromaScaled = np.multiply(ChromaVector,scale[1]['scaleArray'])
        
        scale[1]['likelihood'] = np.sum(ChromaScaled) / NumNotesScale
        
        #print(scale[1]['likelihood'])
    
    maxLikelihood = ['NA',0]
    likelihoods = []
    for item in ScaleTemplates.items():
        if item[1]['likelihood']>maxLikelihood[1]:
            maxLikelihood[0]=item;maxLikelihood[1] = item[1]['likelihood']
        likelihoods.append(item)
        sortedlikelihoods = sorted(likelihoods, key = lambda k:k)
    return(maxLikelihood,sortedlikelihoods)     


def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def VisualizeScaleLikelihoods(filename, dataIndex, likelihoodmethod):
    
    with open('ExtractedFeatures_for12bins!.pkl', 'rb') as f:
        data = pickle.load(f)
    
    data1 = data[filename]
    hpcps = data1[dataIndex]['hpcp']
    hpcpAgg = np.zeros_like(hpcps[0])

    scalelikelihoods = []
    scaletypes = []
    for i in range(len(hpcps)):
        hpcpAgg = hpcpAgg + hpcps[i]
        #print(hpcpAgg)
        if likelihoodmethod == 1:
            maxscalelike, likelihood = maxlikelihood(hpcpAgg)
        elif likelihoodmethod == 2:    
            maxscalelike, likelihood = maxlikelihood2(hpcpAgg)

        framelikelihoods = []
        for j in range(len(likelihood)):
            framelikelihoods.append(likelihood[j][1]['likelihood'])
            scaletypes.append(likelihood[j][0])          
        scalelikelihoods.append(framelikelihoods)
        '''
    for j in range(len(hpcpAgg)):
        hpcpAgg[j] = hpcpAgg[j]/np.sum(hpcpAgg[j])
    print(hpcpAgg)    
    '''
    scaletypes = set(scaletypes)  
    sc = sorted(scaletypes)
    print(sc)
    print(scalelikelihoods[-1])
    MaxLikelihoodNormalized = maxscalelike[1] / np.sum(scalelikelihoods[-1])
    print('Maximum Likeliest Scale of Phrase :' + str(maxscalelike[0][0]) + '    with likeliest : ' + str(MaxLikelihoodNormalized))
    #print(scalelikelihoods
    fig = plt.figure()
    plt.imshow(np.transpose(scalelikelihoods),aspect = 'auto',interpolation = 'nearest',origin = 'lower',cmap = 'magma')
    plt.xlabel('Frame #')
    plt.ylabel('ScaleTypes')
    tick_marks = np.arange(len(sc))
    plt.yticks(tick_marks, sc)
    plt.show()    
        
        
        
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


######## SINGLE FILE PROCESSING - NNLS CHROMA ##############
    
def FeatureExtraction_singleNNLS(fileName, fileDir, params,annotationFile):
    '''
    fileName : input audio file (.mp3)
    fileDir : directory of the audio file
    params : Analysis parameters object
    annotationData : Data from the annotation (exercise.json)
    '''
    
    numBins = params.numBins    
    fs=params.fs
    stepSize = params.hopSize
    keys = {'A':0,'Bb':1,'B':2,'C':3,'C#':4,'D':5,'Eb':6,'E':7,'F':8,'F#':9,'G':10,'Ab':11}
    
    mywindow = np.array(
        [0.001769, 0.015848, 0.043608, 0.084265, 0.136670, 0.199341, 0.270509, 0.348162, 0.430105, 0.514023,
         0.597545, 0.678311, 0.754038, 0.822586, 0.882019, 0.930656, 0.967124, 0.990393, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803,
         0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999803, 0.999650, 0.996856, 0.991283,
         0.982963, 0.971942, 0.958281, 0.942058, 0.923362, 0.902299, 0.878986, 0.853553, 0.826144,
         0.796910, 0.766016, 0.733634, 0.699946, 0.665140, 0.629410, 0.592956, 0.555982, 0.518696,
         0.481304, 0.444018, 0.407044, 0.370590, 0.334860, 0.300054, 0.266366, 0.233984, 0.203090,
         0.173856, 0.146447, 0.121014, 0.097701, 0.076638, 0.057942, 0.041719, 0.028058, 0.017037,
         0.008717, 0.003144, 0.000350])
    
    audio = ess.MonoLoader(filename = os.path.join(fileDir, fileName), sampleRate = fs)()
    audio = ess.DCRemoval()(audio) ##preprocessing / apply DC removal for noisy regions
    audio = ess.EqualLoudness()(audio)
    
    with open(fileDir+annotationFile) as json_file:
        annotationData=json.load(json_file)
    
    partsList = dict()  
           
    for part in annotationData['parts']:
        fileData=initiateData4File(part['name'],fileDir)
        fileData['startTime'] = part['startTime']
        fileData['endTime'] = part['endTime']
        fileData['key'] = part['scaleType'].split(':')[0]
        fileData['scaleType'] = part['scaleType'].split(':')[1]
        
        audio_slice = sliceAudiotoParts(audio,fileData['endTime'],fileData['startTime'],params)
        
        '''
        stepsize, semitones = vamp.collect(audio_slice, 
                                           fs, "nnls-chroma:nnls-chroma", 
                                           parameters = {'chromanormalize':2} , 
                                           output="semitonespectrum", 
                                           step_size=stepSize)["matrix"]
        '''
        
        stepsize, semitones = vamp.collect(audio_slice, 
                                           fs, "nnls-chroma:nnls-chroma", 
                                           output="semitonespectrum", 
                                           step_size=stepSize)["matrix"]
        
        chroma = np.zeros((semitones.shape[0], 12))
        for i in range(semitones.shape[0]):
            tones = semitones[i] * mywindow
            cc = chroma[i]
            for j in range(tones.size):
                cc[j % 12] = cc[j % 12] + tones[j]
            
    
        for key in keys:
            if key == part['scaleType'].split(':')[0]:
                pitch_shift = keys[key]
        pitch_shift = pitch_shift * -1                
        # roll from 'A' based to 'C' based
        chroma = np.roll(chroma, shift=pitch_shift, axis=1)        
            
        fileData['NNLS'] = chroma                    
        
        chromaMean = []
        chromaSTD = []
        for j in range(numBins):
            chromas = [];
            for i in range(len(fileData['NNLS'])):
                chromas.append(fileData['NNLS'][i][j])
            
            chromaMean.append(np.mean(chromas))
            chromaSTD.append(np.std(chromas))
        
        for j in range(numBins):
            chromaMean[j]=chromaMean[j]/np.sum(chromaMean)
            chromaSTD[j]=chromaSTD[j]/np.sum(chromaSTD)

        fileData['mean_NNLS'] = chromaMean
        fileData['std_NNLS'] = chromaSTD
        
        partsList[part['name']] = fileData
        
    fieldnames=['name']
    for i in range(numBins):
        ind=str(i)
        fieldnames.append('mean_NNLS'+ind)
    #for i in range(numBins):
     #   ind=str(i)
      #  fieldnames.append('std_NNLS'+ind)    
    fieldnames.append('scaleType')
    
    dataList=[]
    dataList.append(fieldnames)
    
        
    
    for part in partsList.items(): ##navigate in dictionary
         
        tempList=[] #temporary List to put attributes for each audio slice (data-point)
        
        dataname = part[0] #name of data
        tempList.append(dataname)
        for i in range(numBins): #append mean_HPCP vector bins separately               
            tempList.append(part[1]['mean_NNLS'][i])
        #for i in range(numBins): #append mean_HPCP vector bins separately            
         #   tempList.append(part[1]['std_NNLS'][i])    
        tempList.append(part[1]['scaleType'])    #append scales for classification
        dataList.append(tempList)

    dataList = sorted(dataList, key=lambda x: x[0])
    
    dataListSorted = []
    dataListSorted.append(dataList[-1])
    for i in range(len(dataList)-1):
        dataListSorted.append(dataList[i])
    
    with open(fileDir+'CSVfilefor_singlefile.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataListSorted)    
        
    return partsList    

############### PREDICTIONS ON SCALES EXERCISE PERFORMANCE ######################

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
    
