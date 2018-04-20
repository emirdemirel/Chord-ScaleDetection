import os
import sys
import pickle
import csv
import numpy as np
import essentia.standard as ess
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')
import IPython.display as ipd
import json
from scipy.stats import dirichlet

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
    keys = {'A':0,'Bb':1,'B':2,'C':3,'C#':4,'D':5,'Eb':6,'E':7,'F':8,'F#':9,'G':10,'G#':11}
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

def generateCSV(filename):
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    numBins = data['toprak_dorian'][0]['numBins']    
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

    with open('mycsvfile.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dataList)    
        