import lowLevelFeatures as ll
import numpy as np


class ScaleTuple:
    ### This class contains the scale with maximum energy and its index
    def __init__(self, scaleType, scaleEnergy, keyName):
        self.scaleType = scaleType
        self.scaleEnergy = scaleEnergy
        self.keyName = keyName

class ScaleEstimate(ScaleTuple):
    ### This class contains the scale estimate with its key, maximum energy and its degree
    def __init__(self, scaleKey, scaleType, scaleEnergy):
        self.scaleKey = scaleKey
        ScaleTuple.__init__(self, scaleType, scaleEnergy)


def beatsyncChroma(filename):

    sampleRate = 44100
    stepSize = 2048
    rawchroma = ll.rawChromaFromAudio(filename)
    beats=ll.rnnBeatSegments(filename)
    indices =  map(lambda x: int(float(x) * sampleRate / stepSize), beats.startTimes)
    avg_chroma = np.zeros((np.size(indices), 12))
    for i in range(np.size(indices)):
        if i == 0:
            avg_chroma[i] = sum(rawchroma[:indices[i]])/(indices[i]+1)	  ## pitch-class-wise summation of chroma features
        else:	                                                      ## within the duration of one quarter note (between two consecutive beats)
            avg_chroma[i] = sum(rawchroma[indices[i-1]:indices[i]]) / (indices[i]-indices[i-1]+1)

    return avg_chroma

def winchroma(chroma, win_len):

    window_chroma = np.zeros((chroma.shape[0],12))
    temp = np.concatenate((chroma, np.zeros((win_len-1, 12))), axis=0)
    for i in range(chroma.shape[0]-1):
        window_chroma[i] = sum(temp[i:i+win_len])
        window_chroma[i] = window_chroma[i]/sum(window_chroma[i])
    return window_chroma

def transposeScales(filename):

    ##In this example .txt file, the template scales are 7 major scales : ionian, lydian, harmonic major, lydian augmented, augmented, blues scale, major pentatonic
    text = np.loadtxt(filename, delimiter=',') ##this function transposes all template scale all other keys
    scales = np.zeros((12,text.shape[0], 12))
    for i in range(12):
        for s in range(text.shape[0]):
            scales[i][s] = np.roll((text[s]),i)    ## normalization on the number of notes in a template scale

    return scales

def multiplyTemplates(chroma,scales):

    ##this function multiplies each window frame of chromas with template scales.
    ##output of each row is the product of corresponding scale (ionian, dorian, etc. )
    ##over chroma window frames

    ##input : chroma = windowed chroma vectors
    ##output : scaleEnergy[i][k] = 'i'th array is the presence of i th scale over  windowed chroma vectors

    num_chroma = chroma.shape[0]
    scaledchroma = np.zeros((len(scales),num_chroma, 12))
    scaleEnergy = np.zeros((len(scales),num_chroma))

    for i in range(len(scales)):
        scaledchroma[i]=np.multiply(chroma,scales[i])
        for k in range(num_chroma-1):
            temp_nonzero = []
            for l in range(12):
                if scaledchroma[i][k][l]>0:
                    temp_nonzero.append(scaledchroma[i][k][l])
            M = float(len(temp_nonzero))      #number of notes in a scale
            if M == 0:
		scaleEnergy[i][k] = 0
	    else:
	        norm_factor = (1.0/M)**M   #normalization factor over the number notes in a scale
                scaleEnergy[i][k] = np.prod(temp_nonzero)/norm_factor

    return scaleEnergy


def maxEnergyScale(scales_Energy):

    ##this function returns the index of scale temp (in the order of scales list) that has the maximum value
    ##in each window frame (beat_sync_chromavectors / window_length)

    num_scales, num_chroma = scales_Energy.shape[0], scales_Energy.shape[1]
    scaleType = np.zeros(num_chroma)
    scaleEnergy = np.zeros(num_chroma)
    for k in range (num_chroma):
            scaleType[k] = np.argmax(scales_Energy[:,k])
            scaleEnergy[k] = max(scales_Energy[:,k])


    return ScaleTuple(scaleType,scaleEnergy,0)

def func1(winchroma,Scales):

    ## inputs:
    ## winchroma = windowed chroma vectors
    ## Scales = the Scales matrix that contains all the transpositions of the template scales

    listScaleTuple = []
    num_pitch = Scales.shape[0]
    data = np.zeros((num_pitch,Scales.shape[1],winchroma.shape[0]))
    for i in range(num_pitch):
        data[i] = multiplyTemplates(winchroma,Scales[i])
        listScaleTuple.append(maxEnergyScale(data[i]))

    dict = {0: 'c', 1: 'csharp', 2: 'd', 3: 'eflat', 4: 'e', 5: 'f', 6: 'fsharp', 7: 'g', 8: 'gsharp', 9: 'a',
            10: 'bflat', 11: 'b'}
    lst = [0,1,2,3,4,5,6,7,8,9,10,11]
    lst = [dict[k] for k in lst]

    for i in range(12):
        listScaleTuple[i].keyName = lst[i]

    sorted_scales = [] ##sort the scale_tuple objects according to their scaleEnergy attribute for each analysis window
    for i in range(len(winchroma)):
        k = sorted(listScaleTuple, key=lambda listScaleTuple: listScaleTuple.scaleEnergy[i], reverse=True)
        sorted_scales.append(k)

    return listScaleTuple, sorted_scales

