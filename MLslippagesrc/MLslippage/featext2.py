# FUNCTION feat(f):
# INPUT: signal f is 2D array-> 1st dim: samples, 2nd dim: different signal profiles
# INPUT EXAMPLE: input force of 200 sample readings of fx,fy,fz will be an input array of (200,3)
# OUTPUT: time and frequency features of f, suming to numfeat features
# OUTPUT EXAMPLE: for the (200,3) input array we get a (numfeat,3) output feature array
#
# Code source: Ioannis Agriomallos
#
# License: BSD 3 clause
#
import time
import numpy as np
import numpy.matlib as npm
from numpy import linalg as la
import math
import scipy.io as sio
from scipy.optimize import curve_fit
from nitime import algorithms as alg
import shutil
import os
from pylab import *
import random
import matplotlib.pyplot as plt
import pywt

threshold = 0.0001
nbins = 3
p = 3
binlims = (-10, 10)

#  TIME DOMAIN FEATURES =============================================================================================
#  Integrated Signal (IS): sumation over 1st dimension


def intsgnl(f):
    return np.array([sum(abs(f), 0)]), time.time()

#  Mean Absolute Value (MAV): 1/N * IS


def meanabs(f):
    return 1./len(f)*intsgnl(f)[0], time.time()

#  MAV SLoPe (MAVSLP): MAV(i+1)-MAV(i)


def meanabsslp(f):
    return meanabs(f[1:, :])[0] - meanabs(f[:-1, :])[0], time.time()

#  Simple Square Integral (SSI): sumation of squares over 1st dimension


def ssi(f):
    return np.array([sum(np.power(f, 2), 0)]), time.time()

#  VARiance (VAR): 1/(N-1) * SSI


def var(f):
    return 1./(len(f)-1) * ssi(f)[0], time.time()

#  Root Mean Square (RMS): sqrt(1/N * SSI)


def rms(f):
    return np.power(1./len(f) * ssi(f)[0], 0.5), time.time()

#  RaNGe (RNG): max(f) - min(f)


def rng(f):
    return np.array([np.amax(f, 0) - np.amin(f, 0)]), time.time()

#  Waveform Length (WL): sumation over (x(n+1)-x(n))


def wavl(f):
    return np.array([sum(abs(f[1:, :]-f[:-1, :]), 0)]), time.time()

#  Zero Crossing (ZC): sumation over {(-x(n+1)*x(n)>=thres)*(|x(n)-x(n+1)|>=thres)}


def zerox(f):
    tmpdiff = abs(f[:-1, :] - f[1:, :]) >= threshold
    tmpmult = -np.multiply(f[1:, :], f[:-1, :]) >= threshold
    return np.array([sum(np.multiply(tmpmult, tmpdiff), 0)]), time.time()

#  Slope Sigh Change (SSC): sumation over {((x(n)-x(n-1))*(x(n)-x(n+1)))>=thres}


def ssc(f):
    tmpd1 = f[1:-1, :] - f[:-2, :]
    tmpd2 = f[1:-1, :] - f[2:, :]
    return np.array([sum(np.multiply(tmpd1, tmpd2) >= threshold, 0)]), time.time()

#  Willison AMPlitude (WAMP): sumation over {(x(n)-x(n-1))>=thres}


def wamp(f):
    tmpd = f[1:, :] - f[:-1, :]
    return np.array([sum(tmpd >= threshold, 0)]), time.time()

#  Histogram of Signal (HS)


def shist(f):
    shist = np.zeros((nbins, f.shape[-1]))
    for i in range(f.shape[-1]):
        tmphist, _ = np.histogram(f[:, i], nbins)
        shist[:, i] = tmphist
    return shist, time.time()

#  EXTRA TIME DOMAIN FEATURES LIKE GOLZ DID IN ICRA2015 =============================================================
#  Integrated Signal Real (ISR): sumation of real values over 1st dimension


def intsgnlr(f):
    return np.array([np.sum(f, 0)]), time.time()

#  Mean Value (MV): 1/N * ISR


def meanv(f):
    return np.array([np.mean(f, 0)]), time.time()

#  Integrated Weighted Signal Real (IWSR): sumation of real values minus their mean, over 1st dimension


def intwsgnlr(f):
    return np.array([sum(f - meanv(f)[0], 0)]), time.time()

#  Standard Deviation (SD): 1/N * sumation over (f-MV)^2


def stdr(f):
    return np.array([np.std(f, 0)]), time.time()

#  MaXimum (MX): max(f)


def mx(f):
    return np.array([np.max(f, 0)]), time.time()

#  RaNGe X (RNGX): number of samples, aka 1st dimension


def rngx(f):
    return np.array([[np.array(f).shape[0] for i in range(np.array(f).shape[1])]]), time.time()

#  RaNGe Y (RNGY): max(f)-min(f), the same as RNG
#     RNG --> implemented above


def rngy(f):
    return rng(f)[0], time.time()

#  MEDian (MED): median(f)


def med(f):
    return np.array([np.median(f, 0)]), time.time()

#  HJORTH Complexity (HJORTH): (sigma_dd/sigma_d)/(sigma_d/sigma),
#  where sigma = stdr(f) = SSI, sigma_d = stdr(f') and sigma_dd = stdr(f'')


def hjorth(f):
    f_d = np.diff(f, axis=0)  # TODO: gradient or diff CHECK!!!!
    f_dd = np.diff(f_d, axis=0)
    sigma = stdr(f)[0] + np.finfo(float).eps
    sigma_d = stdr(f_d)[0] + np.finfo(float).eps
    sigma_dd = stdr(f_dd)[0]
    return (sigma_dd/sigma_d)/((sigma_d/sigma) + np.finfo(float).eps), time.time()

#  Shannon's ENTRopy (SENTR): - sumation over p(f)*log2(p(f)), where p(f) is the probability distribution of f


def sentr(f):
    n_f = f.shape[0]  # length of f
    res = 10.
    if n_f <= 1:
        return 0, time.time()
    #  find the bins for each column of f, after you perform a normalisation
    try:
        counts = [np.bincount(np.abs(np.int_(res * (f[:, i] - np.mean(f[:, i])) / (np.std(f[:, i]) + np.finfo(float).eps)))) for i in range(f.shape[1])]
    except ValueError:
        return np.zeros(1, f.shape[1]), time.time()
    probs = np.array([c / (f.shape[0] * 1.) for c in counts])
    ent = [np.sum(-np.multiply(i, np.log2(i + np.finfo(float).eps))) if np.count_nonzero(i) > 1 else 0 for i in probs]
    return np.array([ent]), time.time()

#  Energy of Signal (SE): sumation of squares over 1st dimension, same as SSI
#     SSI --> implemented above


def se(f):
    return ssi(f)[0], time.time()

#  SKewness of Signal (SSK): (IWSR)/(SD^3)


def ssk(f):
    return np.divide(intwsgnlr(f)[0], (stdr(f)[0]**3 + np.finfo(float).eps)), time.time()

#  AutoCORreLation (ACORL): (sumation{i=1:n-k}{(f_i - MV)(f_i+k - MV))}/(sumation{i=1:n-1}{(f_i - MV)^2})


def acorl(f):
    result = np.array([np.correlate(f[:, i], f[:, i], mode='full') for i in range(f.shape[1])]).transpose()
    return result[result.shape[0] / 2:], time.time()

#  FREQUENCY DOMAIN FEATURES LIKE GOLZ DID IN ICRA2015 ==============================================================
#  Frequency of Fit to Amplitude of Fourier (FFAF): a+b*cos(w*t)+c*sin(w*t)


def func(t, a, b, c, w):
    return a + b * np.cos(w * t) + c * np.sin(w * t)


def handle_curve_fit(fn, x, y):
    try:
        return curve_fit(fn, x, y)
    except RuntimeError:
        return np.zeros(4), np.zeros((4, 4))


def ffaf(aFFT):
    FFTsz = aFFT.shape[0]
    xdata = np.array(range(FFTsz))
    out = [handle_curve_fit(func, xdata, aFFT[:, i])[0] for i in range(aFFT.shape[1])]
    popt = [i[3] for i in out]
    return np.array([popt]), time.time()  # return w, frequency of fitted curve!!

#  FREQUENCY DOMAIN FEATURES ========================================================================================
#  AutoRegressive COefficients


def arco(f):
    if len(f.shape) <= 1:
        arco, _ = alg.AR_est_YW(f, p)
    else:
        arco = np.array([alg.AR_est_YW(f[:, i], p)[0] if sum(abs(f[:, i])) > 1e-5 else np.zeros(p) for i in range(f.shape[-1])])
    #  print f.shape, arco.shape
    return arco.transpose(), time.time()

#  MeaN, MeDian, Modified MeaN & Modified MeDian Frequencies


def mf(f):
    FFT = np.fft.rfft(f, axis=0)  # FFT of signal
    RF = np.real(FFT)  # Real part of FFT
    IF = np.imag(FFT)  # Imaginary part of FFT
    F = np.abs(FFT)  # Magnitude of spectrum
    # AF = np.sqrt(np.power(RF,2)+np.power(IF,2))/FFT.shape[0]  # Amplitude of FFT
    AF = np.abs(FFT)
    PF = np.arctan(np.divide(IF, RF + np.finfo(float).eps))  # Phase of FFT
    PF = np.power(F, 2)                # Power     of spectrum
    PF[1:-1] = 2 * PF[1:-1]
    sumF = 0.5 * sum(F[1:], axis=0)
    sumPF = 0.5 * sum(PF[1:], axis=0)
    if len(F.shape) <= 1:
        F = F[:, np.newaxis]
        PF = PF[:, np.newaxis]
    freq = npm.repmat(np.array(range(F.shape[0]))[:, np.newaxis], 1, F.shape[-1])
    MDF = np.array([next(i for i in range(1, len(freq) + 1) if sum(PF[1:i+1, j], axis=0) >= sumPF[j]) for j in range(PF.shape[-1])])
    MMDF = np.array([next(i for i in range(1, len(freq) + 1) if sum(F[1:i+1, j], axis=0) >= sumF[j]) for j in range(F.shape[-1])])
    sumPF[sumPF == 0] = 1.
    sumF[sumF == 0] = 1.
    MNF = sum(np.divide(np.multiply(PF[1:], freq[1:]), sumPF), axis=0)
    MMNF = sum(np.divide(np.multiply(F[1:], freq[1:]), sumF), axis=0)
    out = np.concatenate((np.array([MNF, MDF, MMNF, MMDF]), RF, IF, F, AF, PF), axis=0)
    return out, np.array([MNF, MDF, MMNF, MMDF]), RF, IF, F, AF, PF, time.time()

#  FEATURE EXTRACTION ===============================================================================================


def feat(f, havelabel=0, featlabel=0, magnFFT=0, featall=0, returntime=0):
    t = time.time()
    if havelabel:
        w = f[:, :-1]
        # print w.shape
    else:
        w = f
        # print w.shape
    # ######################################### Feature Names ###########################################################
    # ###################################################################################################################
    # #  features:                                                                                  ||      if         ##
    # #  |----------> time domain      :                                                            || samples = 1024  ##
    # #  |------------|---> phinyomark : 11+3{shist} -----------------------------> = 14+0.0samples ||             14  ##
    # #  |------------|---> golz       : 10+samples{acrol} -----------------------> = 10+1.0samples ||           1034  ##
    # #  |----------> frequency domain :                                                                               ##
    # #  |------------|---> phinyomark : 3{arco}+4{mf}+3(samples/2+1){RF,IF} -----> =  9+1.0samples ||           1033  ##
    # #  |------------|---> golz       : 1{ffaf}+2(samples/2+1){AF,PF} -----------> =  3+1.0samples ||           1027  ##
    # #  |------------|--------|-------alltogether--------------------------------> = 36+3.5samples || numfeat = 3108  ##
    # ###################################################################################################################
    if featlabel == 0:  # use both time and frequency domain features
        MF = mf(w)
        if featall == 1 or featall == 0:
            featlist1 = [intsgnl(w), meanabs(w), meanabsslp(w), ssi(w), var(w), rms(w), rng(w), wavl(w),
                         zerox(w), ssc(w), wamp(w), shist(w), arco(w), (np.concatenate(MF[1:4], axis=0), MF[-1])]
            featlist1 = [np.array(i)[np.newaxis, :] for i in featlist1]
            # print [i.shape for i in featlist1]
            feat1 = np.concatenate(featlist1, axis=0)
        # redundant feats: rngy same as rng, se same as ssi
        if featall == 2 or featall == 0:
            featlist2 = [meanv(w), stdr(w), mx(w), rngx(w), rngy(w), med(w), hjorth(w), sentr(w), se(w), ssk(w), acorl(w), (np.concatenate(MF[5:-1], axis=0), MF[-1])]
            # , ffaf(MF[5])]
            featlist2 = [np.array(i)[np.newaxis, :] for i in featlist2]
            # print [i.shape for i in featlist2]
            feat2 = np.concatenate(featlist2, axis=0)
    elif featlabel == 1:  # use only time domain features
        if featall == 1 or featall == 0:
            featl1 = [intsgnl(w), meanabs(w), meanabsslp(w), ssi(w), var(w), rms(w), rng(w), wavl(w), zerox(w), ssc(w), wamp(w), shist(w)]
            featl1 = [np.array(i)[np.newaxis, :] for i in featl1]
            feat1 = np.concatenate(featl1, axis=0)
        if featall == 2 or featall == 0:
            featl2 = [meanv(w), stdr(w), mx(w), rngx(w), rngy(w), med(w), hjorth(w), sentr(w), se(w), ssk(w), acorl(w)]
            featl2 = [np.array(i)[np.newaxis, :] for i in featl2]
            feat2 = np.concatenate(featl2, axis=0)
    elif featlabel == 2:  # use only frequency domain features
        MF = mf(w)
        if featall == 1 or featall == 0:
            featl1 = [arco(w), (np.concatenate(MF[1:4], axis=0), MF[-1])]
            featl1 = [np.array(i)[np.newaxis, :] for i in featl1]
            feat1 = np.concatenate(featl1, axis=0)
        if featall == 2 or featall == 0:
            featl2 = [(np.concatenate(MF[5:-1], axis=0), MF[-1])]  # , ffaf(MF[5])]
            featl2 = [np.array(i)[np.newaxis, :] for i in featl2]
            feat2 = np.concatenate(featl2, axis=0)
    elif featlabel == 3:  # use only FFT
        MF = mf(w)
        if featall == 1 or featall == 0:
            if magnFFT == 0:  # FFT in real and imaginary part format
                feat1 = np.array((np.concatenate(MF[2:4], axis=0), MF[-1]))[np.newaxis, :]
            else:            # FFT in magnitude format
                feat1 = np.array((MF[4], MF[-1]))[np.newaxis, :]
        if featall == 2 or featall == 0:
            feat2 = np.zeros((0, MF[4].shape[1]))
    if featall == 0:  # use all features
        feat = np.concatenate((feat1, feat2), axis=0)
    elif featall == 1:
        feat = feat1
    elif featall == 2:
        feat = feat2
    if havelabel == 0:
        if returntime == 0:
            return feat[0]
        else:
            return feat, time.time()
    else:
        # assume last column's last element is label
        l = np.ones((feat.shape[0], 1))*f[-1, -1]
        if returntime == 0:
            # print feat.shape, l.shape, feat[:,0].shape
            return np.concatenate((feat[:, 0][:, np.newaxis], l), axis=1)
        else:
            return np.concatenate((feat, l), axis=1), time.time()
