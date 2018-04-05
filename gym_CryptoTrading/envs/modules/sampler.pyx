#!python

#cython: boundscheck=False
#cython: wraparound=False

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
import poloniexdb 
import pywt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

##################################################################

cdef DTYPE_t sum_array(np.ndarray[DTYPE_t, ndim = 1] data):
    cdef int samples = data.shape[0]
    cdef DTYPE_t total = 0.0
    for i in range(samples):
        total = total + data[i]
    return total

##################################################################

cpdef np.ndarray[DTYPE_t, ndim = 1] diff(np.ndarray[DTYPE_t, ndim = 1] data):
    cdef int samples = data.shape[0]
    return data[1:] - data[0:samples - 1]

cpdef np.ndarray[DTYPE_t, ndim = 2] changeTimeWindow(np.ndarray[DTYPE_t, ndim = 1] data, int timeSteps):
    cdef int samples = data.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 2] processed = np.zeros((samples - timeSteps + 1, timeSteps))
    for i in range(samples - timeSteps + 1):
        processed[i][:] = data[i:i + timeSteps].copy()
    return processed

cpdef np.ndarray[DTYPE_t, ndim = 1] undoTimeWindow(np.ndarray[DTYPE_t, ndim = 2] data):
    cdef int samples = data.shape[0]
    cdef int timeSteps = data.shape[1]
    cdef np.ndarray[DTYPE_t, ndim = 1] processed = np.zeros((samples + timeSteps - 1))
    processed[0:timeSteps] = data[0].copy()
    processed[timeSteps:] = data[1:][timeSteps].copy()
    return processed

cpdef localScale(np.ndarray[DTYPE_t, ndim = 1] data,
                 int timeSteps):
    cdef np.ndarray[DTYPE_t, ndim = 2] processed = changeTimeWindow(data, timeSteps)
    cdef int samples = processed.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 2] scaled = np.zeros((samples, timeSteps))
    cdef np.ndarray[DTYPE_t, ndim = 2] weights = np.zeros((samples, 2))
    for i in range(samples):
        localMin = processed[i,:].min()
        localMax = processed[i,:].max()
        weights[i][0] = localMin
        weights[i][1] = localMax
        scaled[i,:] = (processed[i,:] - localMin)/(localMax - localMin)
    return scaled, weights

cpdef np.ndarray[DTYPE_t, ndim = 2] undoLocalScale(np.ndarray[DTYPE_t, ndim = 2] data,
                                                   np.ndarray[DTYPE_t, ndim = 2] weights):
    cdef int samples = data.shape[0]
    cdef int timeSteps = data.shape[1]
    cdef np.ndarray[DTYPE_t, ndim = 2] rescaled = np.zeros((samples, timeSteps))
    for i in range(samples):
        rescaled[i,:] = data[i,:]*(weights[i][1] - weights[i][0]) + weights[i][0]
    return rescaled

##################################################################

cdef np.ndarray[DTYPE_t, ndim = 1] MA(np.ndarray[DTYPE_t, ndim = 1] data, int timeWindow):
    cdef int samples = data.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] ma = np.zeros(samples - timeWindow + 1, dtype = DTYPE)
    for i in range(samples - timeWindow + 1):
        ma[i] = sum_array(data[i: i + timeWindow])/timeWindow
    return ma

cdef np.ndarray[DTYPE_t, ndim = 1] EMA(np.ndarray[DTYPE_t, ndim = 1] data, int timeWindow):
    cdef int samples = data.shape[0]
    cdef DTYPE_t alpha = 2.0/(timeWindow + 1)
    cdef np.ndarray[DTYPE_t, ndim = 1] S = np.zeros(samples - timeWindow + 1, dtype = DTYPE)
    S[0] = sum_array(data[0:timeWindow])/timeWindow
    for i in range(1, samples - timeWindow + 1):
        S[i] = alpha*data[i + timeWindow - 1] + (1 - alpha)*S[i - 1]
    return S

cdef np.ndarray[DTYPE_t, ndim = 2] MACD(np.ndarray[DTYPE_t, ndim=1] data):
    cdef np.ndarray[DTYPE_t, ndim=1] macd = EMA(data, 12)[14:] - EMA(data, 26)
    cdef np.ndarray[DTYPE_t, ndim=1] Signal = EMA(macd, 9)
    cdef np.ndarray[DTYPE_t, ndim=2] output = np.stack([macd[8:],Signal], axis = 1)
    return output

cdef np.ndarray[DTYPE_t, ndim = 1] MTM(np.ndarray[DTYPE_t, ndim=1] data, int lookBack):
    cdef int samples = data.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] mtm = np.zeros(samples - lookBack, dtype = DTYPE)
    mtm[:] = data[lookBack:] - data[:samples-lookBack]
    return mtm

cdef np.ndarray[DTYPE_t, ndim = 1] ROC(np.ndarray[DTYPE_t, ndim=1] data, int lookBack):
    cdef int samples = data.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] roc = np.zeros(samples - lookBack, dtype = DTYPE)
    roc[:] = (data[lookBack:] - data[:samples-lookBack])/data[:samples-lookBack]
    return roc

cdef np.ndarray[DTYPE_t, ndim = 1] CCI(np.ndarray[DTYPE_t, ndim=1] high,
                                       np.ndarray[DTYPE_t, ndim=1] low,
                                       np.ndarray[DTYPE_t, ndim=1] close,
                                       int timeWindow):
    cdef np.ndarray[DTYPE_t, ndim=1] typicalPrice = (high + low + close)/3
    cdef np.ndarray[DTYPE_t, ndim=1] SMA = MA(typicalPrice, timeWindow)
    cdef DTYPE_t std = np.std(typicalPrice[timeWindow - 1:])
    cdef np.ndarray[DTYPE_t, ndim=1] cci = (typicalPrice[timeWindow - 1:] - SMA)/(0.015*std)
    return cci

cdef np.ndarray[DTYPE_t, ndim = 2] BOLL(np.ndarray[DTYPE_t, ndim = 1] data, int timeWindow):
    cdef np.ndarray[DTYPE_t, ndim = 1] SMA = MA(data, timeWindow)
    cdef int length = SMA.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] devs = np.zeros(length, dtype = DTYPE)
    for i in range(length):
        devs[i] = 2.0*np.std(data[i: i + timeWindow])
    return np.stack([SMA + devs, SMA - devs, SMA], axis = 1)

cdef np.ndarray[DTYPE_t, ndim = 1] ATR(np.ndarray[DTYPE_t, ndim = 1] high,
                                       np.ndarray[DTYPE_t, ndim = 1] low,
                                       np.ndarray[DTYPE_t, ndim = 1] Open,
                                       int timeWindow):
    cdef int samples = Open.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] atr = np.zeros(samples, dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] trueRange = np.stack([high - low,
                                                             np.absolute(high - Open),
                                                             np.absolute(low - Open)], axis=1).max(axis = 1)
    atr[0] = sum_array(trueRange[0:timeWindow])/timeWindow
    atr[1:] = (atr[0:samples - 1]*(timeWindow - 1) + trueRange[1:])/timeWindow
    return atr

cdef np.ndarray[DTYPE_t, ndim = 1] SMI(np.ndarray[DTYPE_t, ndim = 1] high,
                                       np.ndarray[DTYPE_t, ndim = 1] low,
                                       np.ndarray[DTYPE_t, ndim = 1] close,
                                       int timeWindow):
    cdef int samples = close.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] centers = np.zeros(samples - timeWindow + 1, dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] differences = np.zeros(samples - timeWindow + 1, dtype = DTYPE)
    cdef DTYPE_t lmax, lmin
    for i in range(centers.shape[0]):
        lmax = high[i: i + timeWindow].max()
        lmin = low[i: i + timeWindow].min()
        centers[i] = (lmax + lmin)/2
        differences[i] = lmax - lmin
    cdef np.ndarray[DTYPE_t, ndim = 1] h = close[timeWindow - 1:] - centers[:]
    cdef np.ndarray[DTYPE_t, ndim = 1] hAux1 = EMA(h,3)
    cdef np.ndarray[DTYPE_t, ndim = 1] hAux2 = EMA(hAux1,3)
    cdef np.ndarray[DTYPE_t, ndim = 1] diffAux1 = EMA(differences,3)
    cdef np.ndarray[DTYPE_t, ndim = 1] diffAux2 = EMA(diffAux1,3)/2.0
    return 100.0*hAux2/diffAux2

cdef np.ndarray[DTYPE_t, ndim = 1] WVAD(np.ndarray[DTYPE_t, ndim = 1] high,
                                        np.ndarray[DTYPE_t, ndim = 1] low,
                                        np.ndarray[DTYPE_t, ndim = 1] Open,
                                        np.ndarray[DTYPE_t, ndim = 1] close):
    cdef int samples = close.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] trueRangeHigh = np.stack([high, Open], axis=1).max(axis=1)
    cdef np.ndarray[DTYPE_t, ndim = 1] trueRangeLow = np.stack([low, Open], axis=1).min(axis=1)
    cdef np.ndarray[DTYPE_t, ndim = 1] ad = np.zeros(samples, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] wvad = np.zeros(samples, dtype=DTYPE)
    for i in range(samples):
        if close[i] > Open[i]:
            ad[i] = close[i] - trueRangeLow[i]
        elif close[i] < Open[i]:
            ad[i] = close[i] - trueRangeHigh[i]
        else:
            ad[i] = 0.0

    wvad[0] = ad[0]
    wvad[1:] = wvad[0: samples - 1] + ad[1:]
    return wvad

##################################################################

cpdef np.ndarray[DTYPE_t, ndim = 3] sample(int size, 
                                           str currencyPair, 
                                           int period,
                                           int timeSteps,
                                           list variables, 
                                           list technicalIndicators,
                                           str wavelet,
                                           int level):
    cdef int margin = 0
    cdef int auxSize = ((size + timeSteps - 1)//(2**level) + 1)*(2**level)
    for entry in technicalIndicators:
        if entry[0] == "MA":
            if margin < entry[1] - 1:
                margin = entry[1] - 1
        elif entry[0] == "EMA":
            if margin < entry[1] - 1:
                margin = entry[1] - 1
        elif entry[0] == "MACD":
            if margin < 33:
                margin = 33
        elif entry[0] == "MTM":
            if margin < entry[1]:
                margin = entry[1]
        elif entry[0] == "ROC":
            if margin < entry[1]:
                margin = entry[1]
        elif entry[0] == "CCI":
            if margin < entry[1] - 1:
                margin = entry[1] - 1
        elif entry[0] == "BOLL":
            if margin < entry[1] - 1:
                margin = entry[1] - 1
        elif entry[0] == "SMI":
            if margin < entry[1] + 3:
                margin = entry[1] + 3
        
    data = poloniexdb.getData(currencyPair, 
                              period, 
                              ["high", "low", "open", "close", "volume", "weightedAverage"], 
                              start = None, 
                              end = 'last', 
                              steps = auxSize + margin)
    
    output = []
    for var in variables:
        output.append(data[var])
    for entry in technicalIndicators:
        if entry[0] == "MA":
            output.append(MA(data[entry[2]], entry[1]))
        elif entry[0] == "EMA":
            output.append(EMA(data[entry[2]], entry[1]))
        elif entry[0] == "MACD":
            Macd = MACD(data[entry[1]]).copy()
            output.append(Macd[:,0])
            output.append(Macd[:,1])
        elif entry[0] == "MTM":
            output.append(MTM(data[entry[2]], entry[1]))
        elif entry[0] == "ROC":
            output.append(ROC(data[entry[2]], entry[1]))
        elif entry[0] == "CCI":
            output.append(CCI(data['high'], data['low'], data['close'], entry[1]))
        elif entry[0] == "BOLL":
            Boll = BOLL(data[entry[2]], entry[1]).copy()
            output.append(Boll[:,0])
            output.append(Boll[:,1])
            output.append(Boll[:,2])
        elif entry[0] == "ATR":
            output.append(ATR(data['high'], data['low'], data['open'], entry[1]))
        elif entry[0] == "SMI":
            output.append(SMI(data['high'], data['low'], data['close'], entry[1]))      
        elif entry[0] == "WVAD":
            output.append(WVAD(data['high'], data['low'], data['open'], data['close']))
    cdef int features = len(output)
    cdef int lenght = 0 
    
    if level > 0:
        decomposed = []
        for i in range(features):
            length = output[i].shape[0]
            auxDecomposed = pywt.swt(output[i][length - auxSize:], wavelet, level = level)
            decomposed.append(auxDecomposed[0][0])
            decomposed.append(auxDecomposed[0][1])
            for lvl in auxDecomposed[1:]:
                decomposed.append(lvl[1])  
        features = len(decomposed)
        res = []
        for i in range(features):
            length = decomposed[i].shape[0]
            auxShifted = changeTimeWindow(decomposed[i][length - size - timeSteps + 1:], timeSteps).copy()
            res.append(auxShifted)
    else:
        res = []
        for i in range(features):
            length = output[i].shape[0]
            auxShifted = changeTimeWindow(output[i][length - size - timeSteps + 1:], timeSteps).copy()
            res.append(auxShifted)
    return np.stack(res, axis = 2)
        
    

cpdef trainingSample(int size, 
                     str currencyPair, 
                     int period,
                     int timeSteps,
                     list variables, 
                     list technicalIndicators,
                     str target,
                     str wavelet,
                     int level):
    cdef int margin = 0
    cdef int auxSize = ((size + timeSteps - 1)//(2**level) + 1)*(2**level)
    for entry in technicalIndicators:
        if entry[0] == "MA":
            if margin < entry[1] - 1:
                margin = entry[1] - 1
        elif entry[0] == "EMA":
            if margin < entry[1] - 1:
                margin = entry[1] - 1
        elif entry[0] == "MACD":
            if margin < 33:
                margin = 33
        elif entry[0] == "MTM":
            if margin < entry[1]:
                margin = entry[1]
        elif entry[0] == "ROC":
            if margin < entry[1]:
                margin = entry[1]
        elif entry[0] == "CCI":
            if margin < entry[1] - 1:
                margin = entry[1] - 1
        elif entry[0] == "BOLL":
            if margin < entry[1] - 1:
                margin = entry[1] - 1
        elif entry[0] == "SMI":
            if margin < entry[1] + 3:
                margin = entry[1] + 3
        
    data = poloniexdb.getData(currencyPair, 
                              period, 
                              ["high", "low", "open", "close", "volume", "weightedAverage"], 
                              start = None, 
                              end = 'last', 
                              steps = auxSize + margin + 1)
    
    output = []
    for var in variables:
        output.append(data[var])
    for entry in technicalIndicators:
        if entry[0] == "MA":
            output.append(MA(data[entry[2]], entry[1]))
        elif entry[0] == "EMA":
            output.append(EMA(data[entry[2]], entry[1]))
        elif entry[0] == "MACD":
            Macd = MACD(data[entry[1]]).copy()
            output.append(Macd[:,0])
            output.append(Macd[:,1])
        elif entry[0] == "MTM":
            output.append(MTM(data[entry[2]], entry[1]))
        elif entry[0] == "ROC":
            output.append(ROC(data[entry[2]], entry[1]))
        elif entry[0] == "CCI":
            output.append(CCI(data['high'], data['low'], data['close'], entry[1]))
        elif entry[0] == "BOLL":
            Boll = BOLL(data[entry[2]], entry[1]).copy()
            output.append(Boll[:,0])
            output.append(Boll[:,1])
            output.append(Boll[:,2])
        elif entry[0] == "ATR":
            output.append(ATR(data['high'], data['low'], data['open'], entry[1]))
        elif entry[0] == "SMI":
            output.append(SMI(data['high'], data['low'], data['close'], entry[1]))      
        elif entry[0] == "WVAD":
            output.append(WVAD(data['high'], data['low'], data['open'], data['close']))
    cdef int features = len(output)
    cdef int lenght = 0
    decomposed = []
    for i in range(features):
        length = output[i].shape[0]
        auxDecomposed = pywt.swt(output[i][length - auxSize - 1: length - 1], wavelet, level = level)
        decomposed.append(auxDecomposed[0][0])
        decomposed.append(auxDecomposed[0][1])
        for lvl in auxDecomposed[1:]:
            decomposed.append(lvl[1])            
    cdef int newFeatures = len(decomposed) 
    res = []
    for i in range(newFeatures):
        length = decomposed[i].shape[0]
        auxShifted = changeTimeWindow(decomposed[i][length - size - timeSteps + 1:], timeSteps).copy()
        res.append(auxShifted)
        
    targetSample = data[target][auxSize + margin - size + 1:]
    return np.stack(res, axis = 2), targetSample.reshape((targetSample.shape[0],1))