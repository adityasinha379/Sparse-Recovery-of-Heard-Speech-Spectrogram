import h5py
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import librosa
import os
import scipy.signal as sig


def lagData(data, lags):
    '''
    Inputs:
    data: time x feature, (resp or spec)
    lags: list/1D array of lag indices
        Should be negative for resp, positive for spec
    Outputs:
    dataLag: time x features*len(lags)
    '''
    dataLag = np.zeros((data.shape[0],
                        data.shape[1]*len(lags)))
    nChan = data.shape[1]
    for ii in range(len(lags)):
        # Shift the data matrix
        lag = lags[ii]
        data_shift = np.roll(data,lag,axis=0)
        # Zero out the values that rolled over
        if lag<0:
            data_shift[lag:,:] = 0
        else:
            data_shift[:lag,:] = 0
        # Put this lagged data in the full matrix
        dataLag[:,ii*nChan:(ii+1)*nChan] = data_shift

    return dataLag

def convertHDF5(filenameIn, filenameOut):

    dataIn = h5py.File(filenameIn)
    dataOut = h5py.File(filenameOut)
    
    fsfinal = 100
    for group in dataIn:
        if group=='SS':
            fsorig = 11025
        else:
            fsorig = 24000
        resp = dataIn[group]['resp'][:,:]
        stim = dataIn[group]['stim'][:]
        spec = librosa.feature.melspectrogram(
                    y=stim, sr=fsorig, power=2,n_fft=2048,
                    hop_length=int(fsorig/fsfinal), n_mels=64)
        if spec.shape[1] != resp.shape[1]:
            spec = spec[:,:resp.shape[1]]
        dataOut.create_dataset(group+'/resp', data=resp)
        dataOut.create_dataset(group+'/spec', data=spec)
    dataIn.close()
    dataOut.close()
    return
    
def getRecFilt(stimTrain, respTrain, lags=np.arange(26)):
    '''
    Input:
    StimTrain: frequency x time, the TF representation of the training stimulus
    respTrain: time x channel, the neural responses to the training stimulus
    lags: are the time delays of the resp used for the estimation
        Default: 0 to 25
    Output:
    g: reconstruction filters
    '''
    respLag = lagResp(respTrain, -lags)
    RR = respLag.T @ respLag
    RS = stimTrain @ respLag
    # Normalize RR
    U, S, Vt = svd(RR)
    S_norm = S/sum(S)
    for ss in range(len(S_norm)):
        if sum(S_norm[:ss])>0.99:
            break
    # Take inverse of sigma-normalized RR
    S = 1./S
    S[ss:] = 0
    RR_inv = Vt.T @ np.diag(S) @ U.T
    # Calculate decoder g
    # g: channel*len(lags) x frequency
    g = RR_inv @ RS.T

    return g

def getSTRF(stimTrain, respTrain, lags=np.arange(26), lam=1e3):
    '''
    Input:
    StimTrain: frequency x time, the TF representation of the training stimulus
    respTrain: channel x time, the neural responses to the training stimulus
    lags: are the time delays of the resp used for the estimation
        Default: 0 to 25
    Output:
    g: STRF
    '''
    # stimLag: time x freq*len(lags)
    stimLag = lagData(stimTrain.T, lags)
    RR = stimLag.T @ stimLag
    RS = respTrain @ stimLag
    # Normalize RR
    U, S, Vt = svd(RR)
    S_norm = S/sum(S)
    for ss in range(len(S_norm)):
        if sum(S_norm[:ss])>0.99:
            break
    # Take inverse of sigma-normalized RR
    S = S+lam
    S = 1./S
    S[ss:] = 0
    RR_inv = Vt.T @ np.diag(S) @ U.T
    # Calculate decoder g
    # g: channel*len(lags) x frequency
    g = RR_inv @ RS.T

    return g

def getX(subjID, trial):
    
    filename = 'Data/'+str(subjID).zfill(3)+'.hdf5'
    data = h5py.File(filename)
    
    spec = data[trial]['spec'][:,:]
    # 0-350 ms lags
    specLag = lagData(spec.T, np.arange(36))
    specLag = specLag.T
    
    data.close()
    
    return specLag

def getY(subjID, trial):
    
    filename = 'Data/'+str(subjID).zfill(3)+'.hdf5'
    data = h5py.File(filename)
    
    resp = data[trial]['resp'][:,:]

    data.close()
    resp = filterResp(resp, fc=8)
    
    return resp

def getA(subjID, trial, lam=1e2):
    
    filename = 'Data/'+str(subjID).zfill(3)+'.hdf5'
    data = h5py.File(filename)
    
    resp = data[trial]['resp'][:,:]
    spec = data[trial]['spec'][:,:]
    data.close()
    
    resp = filterResp(resp, fc=8)
    
    strf = getSTRF(spec, resp, lags=np.arange(36), lam=lam)
    strf = strf.T
    
    return strf

def filterResp(resp, fc=8):
    
    # Define filter (resp is fs=100 Hz)
    b = sig.firwin(101, fc, fs=100)
    # Filter response
    respFilt = sig.filtfilt(b=b, a=1, x=resp, axis=1)
    
    return respFilt

def saturate(x, a=5):
    return a*np.tanh(x/a)