#! /usr/bin/env python
"""Mainly Edited for private usage by:  Ioannis Agriomallos
                                        Ioanna Mitsioni
License: BSD 3 clause

============= CURRENT CODE USAGE =============
Current code trains MLP Classifiers, to classify force input samples as stable (0) or slip (1)
---- Input
-> Input samples originate from Optoforce and ATI sensors and are 3D (fx,fy,fz) and come into 3 different datasets,
   one training (Optoforce), containing several surfaces as well as slip-stable occurrences,
   one validation (Optoforce), containing 1 surface with slip-stable occurrences on a completely unseen task-setup
   and one testing set acquired from ATI sensor for different normal desired forces, for several low-pass filter cutoff
   frequencies and for both translational and rotational slipping occurrences.
---- Input transformation
-> Several pre-features can be taken from these inputs, but here |f| is kept.
-> Several time and frequency domain features are extracted from pre-feature windows.
  (implemented in 'featext.py') These windows have size w and are shifted by s on each sample
-> Then a feature selection-ranking is performed using MutualVariableInformation
-> Finally PCA is performed to keep a reduced set among the best selected features
---- Training of ML Classifiers
-> Several MLP Classifiers are trained for all combinations of selected featuresets (using the training Optoforce dataset)
---- Results
-> Stats of classification results are kept inside each .npz along with the respective trained model in results* folders
"""
# ############################################## EXAMPLE OF CODE USAGE ################################################
# ############ TRAINING PROCEDURE ##############
# # necessary steps before training
# f,l,fd,member,m1,m2 = data_prep(datafile)                      # read input force and labels
# prefeat = compute_prefeat(f)                                   # compute corresponding prefeatures
# features, labels = feature_extraction(prefeat, member)         # feature extraction from prefeatures
# avg_feat_comp_time(prefeat)                                    # average feature extraction time
# new_labels = label_cleaning(prefeat,labels,member)             # trim labels, around change points
# X,Y,Yn,Xsp,Ysp = computeXY(features,labels,new_labels,m1,m2)   # compute data and labels, trimmed and untrimmed
# surf, surfla = computeXY_persurf(Xsp,Ysp)                      # compute per surface data and labels
# # training
# train_1_surface(surf,surfla)                                   # training of all combinations per 1 surface
# train_2_surface(surf,surfla)                                   # training of all combinations per 2 surfaces
# train_3_surface(surf,surfla)                                   # training of all combinations per 3 surfaces
# train_4_surface(surf,surfla)                                   # training of all combinations per 4 surfaces
# train_5_surface(surf,surfla)                                   # training of all combinations per 5 surfaces
#
# ############ OFFLINE TESTING PROCEDURE ##############
# # generate files with stats
# bargraph_perf_gen1(6)
# bargraph_perf_gen2(6)
# bargraph_perf_gen3(6)
# bargraph_perf_gen4(6)
# bargraph_perf_gen5(6)
# # use the bargraph tool to plot graphs from generated files
# # -left column cross-accuracy (trained on one, tested on all the others),
# # -right column self-accuracy (trained and tested on the same)
# # -each row i represents training only with i surfaces.
# # -each stack represents a training group, each bar represents a subfeatureset(AFFT,FREQ,TIME,BOTH)
# # -blue,green,yellow,red : TP,TN,FN,FP
# plt.figure(figsize=(20,40))
# for i in range(5):
#     make_bargraphs_from_perf(i)
#
# ############ ONLINE TESTING PROCEDURE ##############
# # same necessary steps as in training
# f,l,fd,member,m1,m2 = data_prep(validfile)
# prefeat = compute_prefeat(f)
# features, labels = feature_extraction(prefeat, member, validfeatfile, 'validfeat_')
# new_labels = label_cleaning(prefeat,labels,member)
# X,Y,Yn,Xsp,Ysp = computeXY(features,labels,new_labels,m1,m2,validXYfile,validXYsplitfile)
# surf, surfla = computeXY_persurf(Xsp,Ysp,validsurffile)
#
# ####### TESTING DATA FROM ATI F/T SENSOR TRANSLATIONAL CASE
# prediction('ati_new_fd1.5N_kp3_152Hz_validation.mat')
# ####### TESTING DATA FROM ATI F/T SENSOR ROTATIONAL CASE
# prediction('ati_new_fd1.5N_kp3_152Hz_validation_rot.mat')
import time
start_time = time.time()
from copy import deepcopy, copy
import math
import scipy.io as sio
import shutil
import os, errno
from random import shuffle
import numpy as np
from pylab import *
from featext import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import re
import datetime
import urllib
import tarfile
import zipfile
import joblib
from subprocess import call, check_output
from joblib import Parallel, delayed, Memory
from tempfile import mkdtemp
import copy_reg
import types
import itertools
import glob

def _pickle_method(m):
    """Useful function for successful convertion from directories and lists to numpy arrays"""
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(types.MethodType, _pickle_method)

def ensure_dir(directory):
    """Useful function for creating directory only if not existent"""
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def comb(n,r):
    """Combinations of n objects by r, namely picking r among n possible.
    comb(n,r) = n!/(r!(n-r)!)
    """
    return math.factorial(n)/(math.factorial(r)*math.factorial(n-r))

############ PRE-FEATURES ############
###### DEFINITION
# featnum 0 : sf    = (fx^2+fy^2+fz^2)^0.5
#         1 : ft    = (fx^2+fy^2)^0.5
#         2 : fn    = |fz|
#         3 : ft/fn = (fx^2+fy^2)^0.5/|fz|
# input (nxm) -> keep (nx3) -> compute pre-feature and return (nx1)
def sf(f):
    """Computation of norm (sf) of force (f)"""
    return np.power(np.sum(np.power(f[:,:3],2),axis=1),0.5)
def ft(f):
    """Computation of tangential (ft) of force (f)"""
    return np.power(np.sum(np.power(f[:,:2],2),axis=1),0.5)
def fn(f):
    """Computation of normal (fn) of force (f)"""
    return np.abs(f[:,2])
def ftn(f):
    """Computation of tangential (ft) to normal (fn) ratio of force (f),
    corresponding to the friction cone boundary
    """
    retft = ft(f)
    retfn = fn(f)
    retft[retfn<=1e-2] = 0
    return np.divide(retft,retfn+np.finfo(float).eps)
def lab(f):
    """Label embedded in input f"""
    return np.abs(f[:,-1])

class ml:
    def __init__(self,c):
        ######## TRAINING DEFAULTS
        global cv, scaler, decomp, names, classifiers, download, delete_big_features
        cv = c.cv
        scaler = c.scaler
        decomp = c.decomp
        names = c.names
        classifiers = c.classifiers

        download = c.download                       # Download pre-computed (1) data or compute them all anew (0)
        delete_big_features = c.delete_big_features # Delete (1) or keep (0) computed big-in-size features,
                                             # helping mainly to avoid several computations when recomputing features

        ############ INITIALISATION PARAMETERS ############
        global window, shift, samplesperdataset, havelabel, returntime, \
               featlabel, magnFFT, featall, featparam, numfeat, nfeat
        window, shift = c.window, c.shift
        samplesperdataset = c.samplesperdataset
        havelabel = c.havelabel
        returntime = c.returntime
        featlabel = c.featlabel         # 0: all features, 1: temporal, 2: frequency, 3: FFT only
        magnFFT = c.magnFFT             # 0: FFT in magnitude format, 1: FFT in real and imag format,
        featall = c.featall             # 0: all, 1: feat1 (phinyomark's), 2: feat2 (golz's)
        featparam = [havelabel,featlabel,magnFFT,featall,returntime]
        CV = c.CV                       # cross validation checks
        numfeat = c.numfeat             # number of features to show
        nfeat = c.nfeat                 # number of features to keep
        ###### Initialize necessary names and paths
        global datapath, datafile, validfile, featpath, allfeatpath, prefeatpath,\
               prefeatname, prefeatfile, featname, featfile, validfeatname, validfeatfile,\
               surffile, XYfile, XYsplitfile, respath, toolfile, toolpath, tool
        datapath = c.datapath
        ensure_dir(datapath)
        datafile = c.datafile
        validfile = c.validfile
        featpath = c.featpath
        ensure_dir(featpath)
        allfeatpath = c.allfeatpath
        ensure_dir(allfeatpath)
        prefeatname = c.prefeatname
        prefeatfile = c.prefeatfile
        featname = c.featname
        featfile = c.featfile
        validfeatname = c.validfeatname
        validfeatfile = c.validfeatfile
        surffile = c.surffile
        XYfile = c.XYfile
        XYsplitfile = c.XYsplitfile
        validsurffile = c.validsurffile
        validXYfile = c.validXYfile
        validXYsplitfile = c.validXYsplitfile
        respath = c.respath
        toolfile = c.toolfile
        toolpath = c.toolpath
        tool = c.tool

        ############ Feature Names ###########
        global featnames
        """features:                                                                       ||      if\n"""+\
        """|--> time domain      :                                                         || samples = 1024\n"""+\
        """|----|---> phinyomark : 11+3{shist} --------------------------> = 14+0.0samples ||             14\n"""+\
        """|----|---> golz       : 10+samples{acrol} --------------------> = 10+1.0samples ||           1034\n"""+\
        """|--> frequency domain :\n"""+\
        """|----|---> phinyomark : 3{arco}+4{mf}+2(samples/2+1){RF,IF} --> =  9+1.0samples ||           1033\n"""+\
        """|----|---> golz       : 2(samples/2+1){AF,PF} ----------------> =  2+1.0samples ||           1026\n"""+\
        """|----|----------------|-------alltogether---------------------> = 35+3.0samples || numfeat = 3107"""
        ## Time Domain Phinyomark feats
        featnames = ['IS', 'MAV', 'MAVSLP', 'SSI', 'VAR', 'RMS', 'RNG', 'WAVL', 'ZC', 'SSC', 'WAMP',
                     'HIST_1', 'HIST_2', 'HIST_3']                                                 # 11+3{shist}
        ## Frequency Domain Phinyomark feats
        featnames += ['ARCO_1', 'ARCO_2', 'ARCO_3', 'MNF', 'MDF', 'MMNF', 'MMDF']                  # 3{arco}+4{mf}
        featnames += ['RF_{:03d}'.format(i) for i in range(window/2+1)]                            # samples/2+1{RF}
        featnames += ['IF_{:03d}'.format(i) for i in range(window/2+1)]                            # samples/2+1{IF}
        ## Time Domain Golz feats
        featnames += ['MV', 'STD', 'MAX', 'RNGX', 'RNGY', 'MED', 'HJORTH', 'SENTR', 'SE', 'SSK']   # 10
        featnames += ['ACORL_{:04d}'.format(i) for i in range(window)]                             # samples{acrol}
        ## Frequency Domain Golz feats
        featnames += ['AF_{:03d}'.format(i) for i in range(window/2+1)]                            # samples/2+1{AF}
        featnames += ['PF_{:03d}'.format(i) for i in range(window/2+1)]                            # samples/2+1{PF}

        ############ PREFEATURES #############
        global prefeatfn, prefeatnames, prefeatid
        prefeatfn = np.array([sf,ft,fn,ftn,lab]) # convert to np.array to be easily indexed by a list
        prefeatnames = np.array(['fnorm','ft','fn','ftdivfn','label'])
        prefeatid = [0,4]     # only the prefeatures with corresponding ids will be computed

        ############ SUBFEATURES #############
        global subfeats
        subfeats = ['AFFT','FREQ','TIME','BOTH']

############ Download necessary files ############
def convert_bytes(num):
    """this function will convert bytes to MB.... GB... etc"""
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def file_size(file_path):
    """this function will return the file size"""
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return file_info.st_size

def download_file(datafile, targetlink):
    """Function for checking if targetfile exists, else downloading it from targetlink to targetpath+targetfile"""
    if not os.path.isfile(datafile):
        print 'Necessary ', datafile, ' not here! Downloading...'
        u = urllib.urlopen(targetlink)
        data = u.read()
        print 'Completed downloading ','{:.2f}'.format(len(data)*1./(1024**2)),'MB of ',datafile,'!'
        u.close()
        with open(datafile, "wb") as f :
            f.write(data)
        print 'Necessary ', datafile, ' completed saving!'
    else:
        print 'Necessary ', datafile, ' already here!'
    return file_size(datafile)

def extract_file(source,destination='.'):
    """Decompress source zip, tar or tgz file to destination folder"""
    print "Extracting compressed file..."
    if (source.endswith('tar.gz') or source.endswith('tgz')):
        with tarfile.open(source, 'r:gz' ) as tgz_ref:
            tgz_ref.extractall(destination)
        print "Done!"
    elif (source.endswith('tar')):
        with tarfile.open(source, 'r:' ) as tar_ref:
            tar_ref.extractall(destination)
        print "Done!"
    elif (source.endswith('zip')):
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall(destination)
        print "Done!"
    else:
        print "Unsupported extension for decompressing. Supported extensions are .zip, .tgz, .tar.gz, .tar"

######### Download necessary dataset #############
def download_required_files():
    total_size_of_downloads = 0
    # datafile = datapath+'dataset.npz'
    # validfile = datapath+'validation.mat'
    datalink = 'https://www.dropbox.com/s/j88wmtx1vvpik1m/dataset.npz?dl=1'
    validlink = 'https://www.dropbox.com/s/r8jl57lij28ljrw/validation.mat?dl=1'
    total_size_of_downloads += download_file(datafile, datalink)
    total_size_of_downloads += download_file(validfile, validlink)
    ####### Download bargraph tool if not already downloaded (by Derek Bruening)
    toollink = 'https://github.com/derekbruening/bargraph/archive/rel_4_8.zip'
    # toolfile = datapath+'bargraph.zip'
    # toolpath = datapath+'bargraph-rel_4_8/'
    if not os.path.isdir(toolpath):
        total_size_of_downloads += download_file(toolfile, toollink)
        if os.path.isfile(toolfile):
            extract_file(toolfile,datapath+'.')
    # tool = './'+toolpath+'bargraph.pl'
    call(['chmod','+x',tool]) # make tool executable
    call(['rm',toolfile]) # delete zip file
    ####### Download features and trained models, if not wanting to compute them and not already there
    if download==1:
        featlink = 'https://www.dropbox.com/s/qvk9pcvlir06zse/features_1024_20_10000.npz?dl=1'
        validfeatlink = 'https://www.dropbox.com/s/sghqwifo8rxwbcs/validfeatures_1024_20_10000.npz?dl=1'
        total_size_of_downloads += download_file(featfile, featlink)
        total_size_of_downloads += download_file(validfeatfile, validfeatlink)
        reslink = {}
        reslink[0] = 'https://www.dropbox.com/sh/mib7wk4sfv6eye3/AACUWSOgQjBD9i2sChtNisNKa?dl=1'
        reslink[1] = 'https://www.dropbox.com/sh/y6js9ha585n4zam/AACARvB8krZnC3VPsOjWTaRra?dl=1'
        reslink[2] = 'https://www.dropbox.com/sh/fc9jgi2cs7d0dzg/AADfw42xG0XtiUOYWo7cmmtUa?dl=1'
        reslink[3] = 'https://www.dropbox.com/sh/mx6e7jcxzbcr5s4/AACkVMPatRd2UZfyUkxvP_tLa?dl=1'
        reslink[4] = 'https://www.dropbox.com/sh/88itj3b4nwpe0f1/AACceO9FsZp5w55n7PKlVnWSa?dl=1'
        for i in range(len(reslink)):
            resfold = datapath+'results'+str(i+1)
            if not os.path.isdir(resfold):
                resfile = resfold+'.zip'
                total_size_of_downloads += download_file(resfile, reslink[i]) # download
                extract_file(resfile, resfold) # extract
                call(['rm',resfile]) # delete zip
            else:
                print "Desired trained models for "+str(i+1)+" surface found!"
    print "Downloaded "+convert_bytes(total_size_of_downloads)+" of content in total!"

############ READ THE DATASET ############
def data_prep(datafile,step=1,k=2,printit=True):
    """Prepare dataset, from each of the k fingers for all n surfaces (see fd for details)
    -> datafile : input file either in .npz or in .mat form
    -> step     : increasing sampling step, decreases sampling frequency of input, which is 1KHz initially
    -> k        : number of fingers logging data
    ----- input format ----- either 'fi', 'li', 'fdi', with i in {1,...,k} for each finger
                             or     'f', 'l', 'fd' for a finger
                             corresponding to force, label and details respectively
    <- f,l,fd   : output force, label and details for each experiment in the dataset
    <- member   : how much each dataset is represented,
                  to skip samples effectively and keep dimensions correct
    <- m1, m2   : portion of data belonging to finger1 and finger2
    """
    if printit:
        print "---------------------------- LOADING DATA and COMPUTING NECESSARY STRUCTS ----------------------------"
    if datafile[-3:]=='mat':
        inp = sio.loadmat(datafile,struct_as_record=True)
    elif datafile[-3:]=='npz':
        inp = np.load(datafile)
    else:
        print "Unsupported input file format. Supported types: .npz .mat"
        return -1
    if k==2:
        f1, f2, l1, l2, fd1, fd2 = inp['f1'], inp['f2'], inp['l1'], inp['l2'], inp['fd1'], inp['fd2']
        if printit:
            print 1, '-> f1:', f1.shape, l1.shape, fd1.shape
            print 2, '-> f2:', f2.shape, l2.shape, fd2.shape
        ####### MERGE THE DATASETS
        f = np.concatenate((f1,f2),axis=0)
        l = np.concatenate((l1,l2),axis=0)
        fd = np.concatenate((fd2,fd2),axis=0)
    elif k==1:
        f, l, fd = inp['f'], inp['l'], inp['fd']
    else:
        print "Unsupported number of fingers k. Should be k in {1,2}"
    if printit:
        print 3, '-> f:', f.shape, l.shape, fd.shape
    # membership of each sample, representing its portion in the dataset
    # (first half finger1 and second half finger2)
    member = np.zeros(len(f))
    m1,m2 = len(f)/2, len(f)/2
    member[:m1] = np.ones(m1)*1./m1
    member[-m2:] = np.ones(m2)*1./m2
    if printit:
        print 4, '-> m1,m2:', m1, m2, sum(member[:m1]), sum(member[-m2:])
    ####### MERGE f and l
    while f.ndim>1:
        f = f[:,0]
        l = l[:,0]
    for i in range(len(f)):
        while l[i].ndim<2:
            l[i] = l[i][:,np.newaxis]
    f = np.array([np.concatenate((f[i],l[i]),axis=1) for i in range(len(f))])
    if printit:
        print 5, '-> f=f+l:', f.shape, ":", [fi.shape for fi in f]
    ####### SUBSAMPLING
    # step = 1 # NO SAMPLING
    if step!=1:
        f = np.array([fi[::step,:] for fi in f])
        if printit:
            print 6, '-> fsampled:',f.shape, ":", [fi.shape for fi in f]
    return f,l,fd,member,m1,m2

############ PRE-FEATURES ############
###### DEFINITION
# featnum 0 : sf    = (fx^2+fy^2+fz^2)^0.5
#         1 : ft    = (fx^2+fy^2)^0.5
#         2 : fn    = |fz|
#         3 : ft/fn = (fx^2+fy^2)^0.5/|fz|
# input (nxm) -> keep (nx3) -> compute pre-feature and return (nx1)
###### COMPUTATION
# prefeatfn = np.array([sf,ft,fn,ftn,lab]) # convert to np.array to be easily indexed by a list
# prefeatnames = np.array(['fnorm','ft','fn','ftdivfn','label'])
# prefeatid = [0,4]     # only the prefeatures with corresponding ids will be computed
def compute_prefeat(f,printit=True):
    """Prefeature computation
    -> f       : input force as an i by n by 4 matrix
    <- prefeat : corresponding force profiles
    """
    if printit:
        print "--------------------------------------- COMPUTING PREFEATURES ----------------------------------------"
    prefeat = [np.array([prfn(f[i]) for prfn in prefeatfn[prefeatid]]).transpose() for i in range(len(f))]
    prefeat.append(prefeat[-1][:-1])
    prefeat = np.array(prefeat)[:-1]
    if printit:
        print prefeat.shape,":",[p.shape for p in prefeat]
    return prefeat

############ AVG Computation time of ALL features in secs ############
def avg_feat_comp_time(prefeat,printit=True):
    """Average computation time for feature extraction
    -> prefeat : desired prefeature input
    """
    if printit:
        print "------------------------------------ AVG FEATURE COMPUTATION TIME ------------------------------------"
    t1 = time.time()
    m = int(ceil(0.2*len(prefeat)))
    # avg over m*100 times
    tmpfeat = [feat(prefeat[k][i:i+window,:2],*featparam) for k in range(m) for i in range(100)]
    if printit:
        print 'Avg feature computation time (millisec): ', (time.time() - t1) / (100 * m) * 1000

############ FEATURE COMPUTATION ############
def tmpfeatfilename(p,name,mode='all'):
    """Filename for feature computation and intermittent saving
    -> p    : prefeat id
    -> name : desired prefix name for tmp filenames
    -> mode : whether keeping whole feature matrix ('all') or sampling rows ('red') to reduce size
    <- corresponding output filename
    """
    if mode == 'all':
        return allfeatpath+name+str(p)+'.pkl.z'
    elif mode == 'red':
        return allfeatpath+name+str(p)+'_red'+str(samplesperdataset)+'.pkl.z'

def feature_extraction(prefeat, member, featfile, name='feat_', printit=True):
    """Computation of all features in parallel or loading if already computed
    -> prefeat          : computed prefeatures
    -> member           : how much each dataset is represented,
                          to skip samples effectively and keep dimensions correct
    -> featfile         : desired final feature filename
    -> name             : desired per dataset feature temporary filenames
    <- features, labels : computed features and corresponding labels
    """
    if printit:
        print "---------------------------------------- FEATURE EXTRACTION ------------------------------------------"
    if os.path.isfile(featfile):
        start_time = time.time()
        features = np.load(featfile)['features']
        labels = np.load(featfile)['labels']
        if printit:
            print("Features FOUND PRECOMPUTED! Feature Loading DONE in: %s seconds " % (time.time() - start_time))
        if delete_big_features:
            for j in glob.glob(allfeatpath+"*"):
                if 'red' not in j:
                    call(['rm',j]) # delete big feature file, after reducing its size to desired
    else:
        start_time = time.time()
        features = []
        labels = []
        for ixp in range(len(prefeat)):
            p = prefeat[ixp]
            now = time.time()
            tmpfn = tmpfeatfilename(ixp,name)
            tmpfnred = tmpfeatfilename(ixp,name,'red')
            if not os.path.isfile(tmpfnred):
                if not os.path.isfile(tmpfn):
                    # Computation of all features in PARALLEL by ALL cores
                    tmp = np.array([Parallel(n_jobs=-1)([delayed(feat) (p[k:k+window],*featparam)
                                                         for k in range(0,len(p)-window,shift)])])
                    with open(tmpfn,'wb') as fo:
                        joblib.dump(tmp,fo)
                    if printit:
                        print 'sample:', ixp, ', time(sec):', '{:.2f}'.format(time.time()-now), tmpfn, ' computing... ', tmp.shape
                else:
                    with open(tmpfn,'rb') as fo:
                        tmp = joblib.load(fo)
                    if printit:
                        print 'sample:', ixp, ', time(sec):', '{:.2f}'.format(time.time()-now), tmpfn, ' already here!', tmp.shape
                # keep less from each feature vector but keep number of samples for each dataset almost equal
                try:
                    tmpskip = int(round(tmp.shape[1]/(member[ixp]*samplesperdataset)))
                except:
                    tmpskip = 1
                if tmpskip == 0:
                    tmpskip = 1
                # Save reduced size features
                tmp = tmp[0,::tmpskip,:,:]
                with open(tmpfnred,'wb') as fo:
                    joblib.dump(tmp,fo)
                if printit:
                    print 'sample:',ixp, ', time(sec):', '{:.2f}'.format(time.time()-now), tmpfnred, tmp.shape
                if delete_big_features:
                    call(['rm',tmpfn]) # delete big feature file, after reducing its size to desired
            else:
                if delete_big_features:
                    call(['rm',tmpfn]) # delete big feature file, since reduced size file exists
        for ixp in range(len(prefeat)):
            if delete_big_features:
                tmpfn = tmpfeatfilename(ixp,name)
                call(['rm',tmpfn]) # delete big feature file if still here for some reason
            tmpfnred = tmpfeatfilename(ixp,name,'red')
            with open(tmpfnred,'rb') as fo:
                tmp = joblib.load(fo)
            if printit:
                print 'sample:', ixp, ', time(sec):', '{:.2f}'.format(time.time()-now), tmpfnred, 'already here!', tmp.shape
            features.append(tmp[:,:,:-1])
            labels.append(tmp[:,0,-1])
        if printit:
            print("Features NOT FOUND PRECOMPUTED! Feature Computation DONE in: %s sec " % (time.time() - start_time))
        features.append(tmp[:-1,:,:-1])
        features = np.array(features)[:-1]
        labels.append(tmp[:-1,0,-1])
        labels = np.array(labels)[:-1]
        if printit:
            print 'features: ',features.shape,[ftmp.shape for ftmp in features]
            print 'labels: ', labels.shape,[l.shape for l in labels]
        np.savez(featfile,features=features,labels=labels)
    if printit:
        print 'features: ', features.shape, ', labels: ', labels.shape
    return features, labels

############ LABEL TRIMMING ############
def label_cleaning(prefeat,labels,member,history=500,printit=True):
    """Keep the purely stable and slip parts of label, thus omitting some samples around sign change points
    -> prefeat    : computed prefeatures
    -> labels     : main structure, where the trimming will be performed around change points
    -> member     : how much each dataset is represented, to skip samples effectively and keep dimensions correct
    -> history    : how much samples to throw away around change points
    <- new_labels : the trimmed labels
    """
    if printit:
        print "----------- KEEPING LABEL's PURE (STABLE, SLIP) PHASE PARTS (TRIMMING AROUND CHANGE POINTS)-----------"
    lbl_approx = []
    for i in range(len(prefeat)):
        tmpd = np.abs(np.diff(prefeat[i][:,-1].astype(int),n=1,axis=0))
        if np.sum(tmpd) > 0:
            tmpind = np.array(range(len(tmpd)))[tmpd > 0]   # find the sign change points
            tmpindrng = []
            for j in range(len(tmpind)):
                length = history                # keep/throw a portion of the signal's length around change points
                tmprng = np.array(range(tmpind[j]-length,tmpind[j]+length))
                tmprng = tmprng[tmprng>=0]      # make sure inside singal's x-range
                tmprng = tmprng[tmprng<prefeat[i].shape[0]]
                tmpindrng += tmprng.tolist()
            tmpindrng = np.array(tmpindrng).flatten()
            tmp_lbl = deepcopy(prefeat[i][:,-1])
            tmp_lbl[tmpindrng] = -1
            lbl_approx.append(tmp_lbl)
        else:
            lbl_approx.append(prefeat[i][:,-1])
    new_labels = deepcopy(labels)
    for ixp in range(len(lbl_approx)):
        p = lbl_approx[ixp]
        tmp = np.array([p[k+window] for k in range(0,len(p)-window,shift)])
        try:
            tmpskip = int(round(tmp.shape[0]/(member[ixp]*samplesperdataset)))
        except:
            tmpskip = 1
        if tmpskip == 0:
            tmpskip = 1
        # Sampling appropriately
        tmp = tmp[::tmpskip]
        if len(tmp) > len(labels[ixp]):
            tmp = tmp[:-1]
        new_labels[ixp] = tmp
    if printit:
        print 'new_labels: ', new_labels.shape
    return new_labels

############ GATHERING into complete arrays ready for FITTING ############
def computeXY(features,labels,new_labels,m1,m2,XYfile,XYsplitfile,printit=True):
    """
    -> features       : computed features as input data
    -> labels         : corresponding labels
    -> new_labels     : labels trimmed around change point
    -> m1, m2         : portion of data belonging to finger1 and finger2
    -> XY[split]file  : desired output filenames
    <- X,Y,Yn,Xsp,Ysp : X corresponds to the data, Y the label, and *sp to the trimmed label's versions
    """
    if printit:
        print "----------------------------- COMPUTING X,Y for CLASSIFIERS' INPUT -----------------------------------"
    if os.path.isfile(XYfile) and os.path.isfile(XYsplitfile):
        X = np.load(XYfile)['X']
        Y = np.load(XYfile)['Y']
        Yn = np.load(XYfile)['Yn']
        Xsp = np.load(XYsplitfile)['X']
        Ysp = np.load(XYsplitfile)['Y']
        if printit:
            print("XY files FOUND PRECOMPUTED!")
    else:
        # gathering features X,Xsp and labels Y,Ysp,Yn into one array each
        ind,X,Xsp,Y,Ysp,Yn = {},{},{},{},{},{}
        ind[2] = range(features.shape[0])                                      # indeces for both fingers
        ind[0] = range(features.shape[0])[:m1]                                 # indeces for finger1
        ind[1] = range(features.shape[0])[-m2:]                                # indeces for finger2
        ind = np.array([i for _,i in ind.items()])                             # convert to array
        for k in range(len(ind)):
            X[k] = features[ind[k]]                                            # input feature matrix
            Y[k] = labels[ind[k]]                                              # output label vector
            Yn[k] = new_labels[ind[k]]                                         # output new_label vector
            if printit:
                print 'Before -> X[',k,']: ',X[k].shape,', Y[',k,']: ',Y[k].shape,', Yn[',k,']: ',Yn[k].shape
            X[k] = np.concatenate(X[k],axis=0)
            Y[k] = np.concatenate(Y[k],axis=0)
            Yn[k] = np.concatenate(Yn[k],axis=0)
            if printit:
                print 'Gathered -> X[',k,']: ',X[k].shape,', Y[',k,']: ',Y[k].shape,', Yn[',k,']: ',Yn[k].shape
            X[k] = np.array([X[k][:,:,i] for i in range(X[k].shape[2])])
            tmp_sampling = int(round(X[k].shape[1]*1./samplesperdataset))
            if tmp_sampling == 0:
                tmp_sampling = 1
            X[k] = X[k][0,::tmp_sampling,:]
            Y[k] = Y[k][::tmp_sampling]
            Yn[k] = Yn[k][::tmp_sampling]
            if printit:
                print 'Gathered, sampled to max ', samplesperdataset, ' -> X[', k,']: ', X[k].shape, \
                                         ', Y[', k, ']: ', Y[k].shape, ', Yn[', k,']: ', Yn[k].shape
            keepind = Yn[k]>=0
            Xsp[k] = X[k][keepind,:]
            Ysp[k] = Yn[k][keepind]
            if printit:
                print 'Split -> Xsp[',k,']: ',Xsp[k].shape,', Ysp[',k,']: ',Ysp[k].shape
        X = np.array([i for _,i in X.items()])
        Xsp = np.array([i for _,i in Xsp.items()])
        Y = np.array([i for _,i in Y.items()])
        Ysp = np.array([i for _,i in Ysp.items()])
        Yn = np.array([i for _,i in Yn.items()])
        np.savez(XYfile,X=X,Y=Y,Yn=Yn)
        np.savez(XYsplitfile, X=Xsp, Y=Ysp)
    if printit:
        print 'X,Y [0,1,2]: ', X[0].shape, Y[0].shape, X[1].shape, Y[1].shape, X[2].shape, Y[2].shape
        print 'Xsp,Ysp [0,1,2]: ', Xsp[0].shape, Ysp[0].shape, Xsp[1].shape, Ysp[1].shape, Xsp[2].shape, Ysp[2].shape
    return X,Y,Yn,Xsp,Ysp

############ Prepare the indeces for each feature ############
def get_feat_id(feat_ind, printit=False):
    """Find the corresponding indeces of the desired features inside feature vector,
    and link them with their names and level of abstraction
    -> feat_ind        : range of indeces
    -> printit         : print output indeces (1) or not (0)
    -> sample_window   : parameter for accurate computation of feature indeces
    <- full_path_id    : indeces of all features
    <- norm_time_feats : indeces of time features
    <- norm_freq_feats : indeces of frequency features
    """
    sample_window = window
    # get the feat inds wrt their source : 3rd level
    norm_time_phin = range(0,14)
    norm_freq_phin = range(norm_time_phin[-1] + 1, norm_time_phin[-1] + 9 + sample_window + 1)
    norm_time_golz = range(norm_freq_phin[-1] + 1, norm_freq_phin[-1] + 10 + sample_window + 1)
    norm_freq_golz = range(norm_time_golz[-1] + 1, norm_time_golz[-1] + 2 + sample_window + 1)
    # get the feat inds wrt their domain : 2nd level
    norm_time_feats = norm_time_phin + norm_time_golz
    norm_freq_feats = norm_freq_phin + norm_freq_golz
    # get the feat inds wrt their prefeat: 1st level
    norm_feats = norm_time_feats + norm_freq_feats

    # get the feat inds wrt their source : 3rd level
    disp = norm_feats[-1]+1
    ftfn_time_phin = range(disp ,disp + 14)
    ftfn_freq_phin = range(ftfn_time_phin[-1] + 1, ftfn_time_phin[-1] + 9 + sample_window + 1)
    ftfn_time_golz = range(ftfn_freq_phin[-1] + 1, ftfn_freq_phin[-1] + 10 + sample_window + 1)
    ftfn_freq_golz = range(ftfn_time_golz[-1] + 1, ftfn_time_golz[-1] + 2 + sample_window + 1)
    # get the feat inds wrt their domain : 2nd level
    ftfn_time_feats = ftfn_time_phin + ftfn_time_golz
    ftfn_freq_feats = ftfn_freq_phin + ftfn_freq_golz
    # get the feat inds wrt their prefeat: 1st level
    ftfn_feats = ftfn_time_feats + ftfn_freq_feats

    # create the final "reference dictionary"
    # 3 np.arrays, id_list[0] = level 1 etc
    id_list = [np.zeros((len(ftfn_feats + norm_feats),1)) for i in range(3)]
    id_list[0][:norm_feats[-1]+1] = 0 # 0 signifies norm / 1 signifies ft/fn
    id_list[0][norm_feats[-1]+1:] = 1

    id_list[1][:norm_time_phin[-1]+1] = 0 # 0 signifies time / 1 signifies freq
    id_list[1][norm_time_phin[-1]+1:norm_freq_phin[-1]+1] = 1
    id_list[1][norm_freq_phin[-1]+1:norm_time_golz[-1]+1] = 0
    id_list[1][norm_time_golz[-1]+1:norm_freq_golz[-1]+1] = 1
    id_list[1][norm_freq_golz[-1]+1:ftfn_time_phin[-1]+1] = 0
    id_list[1][ftfn_time_phin[-1]+1:ftfn_freq_phin[-1]+1] = 1
    id_list[1][ftfn_freq_phin[-1]+1:ftfn_time_golz[-1]+1] = 0
    id_list[1][ftfn_time_golz[-1]+1:] = 1

    id_list[2][:norm_freq_phin[-1]+1] = 0 #0 signifies phinyomark / 1 signifies golz
    id_list[2][norm_freq_phin[-1]+1:norm_freq_golz[-1]+1] = 1
    id_list[2][norm_freq_golz[-1]+1:ftfn_freq_phin[-1]+1] = 0
    id_list[2][ftfn_freq_phin[-1]+1:] = 1

    full_path_id = [np.zeros((len(feat_ind),5)) for i in range(len(feat_ind))]
    freq_path_id = []
    time_path_id = []

    for ind, val in enumerate(feat_ind):
        full_path_id[ind] = [val, id_list[2][val], id_list[1][val], id_list[0][val]]
        if(full_path_id[ind][1]==0):
            lvl3 = 'Phin'
        else:
            lvl3 = 'Golz'
        if(full_path_id[ind][2]==0):
            lvl2 = 'Time'
            time_path_id.append(val)
        else:
            lvl2 = 'Freq'
            freq_path_id.append(val)
        if(full_path_id[ind][3]==0):
            lvl1 = 'Norm'
        else:
            lvl1 = 'Ft/Fn'
        if (printit):
            print(feat_ind[ind],featnames[val%(norm_feats[-1]+1)],lvl3,lvl2,lvl1)
    return(full_path_id,time_path_id,freq_path_id)

def get_feat_names(printit=True):
    """Return a list with all computed feature names"""
    return featnames

def get_feat_ids_from_names(feat_name_list, printit=False):
    """Return a list of indexes corresponding to the given list of feature names"""
    tmpfind = []
    for m in feat_name_list:
        try:
            ti = m.index('_')
        except:
            ti = len(m)+1
        for i in range(len(featnames)):
            if featnames[i][:ti] == m[:ti]:
                tmpfind.append(i)
    if printit:
        print tmpfind
        print np.array(featnames)[tmpfind]
    return tmpfind

############ Surface Splitting ############
def surface_split(data_X, data_Y, n=6, k=2, printit=True):
    """Split input data in k*n equal slices which represent n different surfaces sampled from k fingers.
    Indexes 0:n:(k-1)*n, 1:n:(k-1)*n+1, 2:n:(k-1)*n+2, ... correspond to the same surface (finger1 upto fingerk)
    Assuming k=2, namely 2 fingers case, unless stated differently
    -> data_X, data_Y        : input data and labels, with the convention that data_X contains k*n almost
                               equally sized data, where the n first are acquired from finger1 ...
                               and the n last from fingerk.
    -> n                     : number of different surfaces
    -> k                     : number of fingers logging data
    <- surfaces, surf_labels : corresponding output data and labels
    """
    keep = data_X.shape[0]-np.mod(data_X.shape[0],k*n)
    surfaces_pre = np.array(np.split(data_X[:keep,:],k*n))
    surf_labels_pre = np.array(np.split(data_Y[:keep],k*n))
    surfaces, surf_labels = {},{}
    for i in range(n):
        inds = range(i,k*n,n)
        surfaces[inds[0]] = surfaces_pre[inds[0]]
        surf_labels[inds[0]] = surf_labels_pre[inds[0]]
        for tk in range(k-1):
            surfaces[inds[0]] = np.concatenate((surfaces[inds[0]], surfaces_pre[inds[tk+1]]), axis = 0)
            surf_labels[inds[0]] = np.concatenate((surf_labels[inds[0]], surf_labels_pre[inds[tk+1]]), axis = 0)
    surfaces = np.array([i for _,i in surfaces.items()])
    surf_labels = np.array([i for _,i in surf_labels.items()])
    return surfaces, surf_labels

############ Featureset Splitting ############
def feat_subsets(data,fs_ind,keep_from_fs_ind):
    """returns a splitting per featureset of input features
    -> data                                : input data X
    -> fs_ind                              : prefeature id
    -> keep_ind                            : list of feature indexes to be kept from whole feature vector
    <- X_amfft, X_freq_all, X_time, X_both : split featuresets amplitude of FFT, all time only,
                                                               all frequency only and all features
    """
    ofs = len(keep_from_fs_ind)
    _,tf,ff = get_feat_id(keep_from_fs_ind)
    amfft_inds = []
    temp1 = deepcopy(data)

    for i in keep_from_fs_ind:
        if (featnames[i].startswith('AF')):
            amfft_inds.append(i)

    if (fs_ind == 2):
        ff2 = [ff[i]+ofs for i in range(len(ff))]
        tf2 = [tf[i]+ofs for i in range(len(tf))]
        amfft2 = [amfft_inds[i]+ofs for i in range(len(amfft_inds))]
        freqf = ff2 + ff
        timef = tf2 + tf
        amfft = amfft_inds + amfft2
    else:
        freqf = ff
        timef = tf
        amfft = amfft_inds

    X_amfft = temp1[:,amfft]
    X_time = temp1[:,timef]
    X_freq_all = temp1[:,freqf]
    X_both = data[:,keep_from_fs_ind]
    return X_amfft, X_freq_all, X_time, X_both

############ Prepare the dataset split for each surface ############
def computeXY_persurf(Xsp, Ysp, surffile, keepind=[-1], n=6, k=2, saveload=True, printit=True):
    """returns a split per surface data and label of inputs
    -> Xsp, Ysp     : input data and labels, after having trimmed data around the label's change points
    -> surffile     : desired output's filename for saving
    <- surf, surfla : output data and label, split per surface
    """
    if len(keepind) == 0 or keepind[0] == -1:
        keepind = range(len(featnames))
    if printit:
        print "------------------------ COMPUTING X,Y per surface CLASSIFIERS' INPUT --------------------------------"
    if os.path.isfile(surffile) and saveload:
        surf = np.load(surffile)['surf']       # input array containing computed features for each surface
        surfla = np.load(surffile)['surfla']   # corresponding label
    else:
        surf, surfla = [], []
        for i in range(len(prefeatid)-1): # for each featureset (corresponding to each prefeature, here only |f|)
            surf1, surfla1 = surface_split(Xsp[2], Ysp[2], n, k, printit)
            tmpsurf = deepcopy(surf1)
            tmpsurfla = deepcopy(surfla1)
            tmpsurfsubfeat = []
            for j in range(tmpsurf.shape[0]+1): # for each surface
                if (printit):
                    print i,j,surf1.shape
                if j == tmpsurf.shape[0]:
                    # ommit a sample for converting to array
                    tmpsurfsubfeat.append(feat_subsets(tmpsurf[j-1,:-1,:],i,keepind))
                else:
                    # keep all subfeaturesets
                    tmpsurfsubfeat.append(feat_subsets(tmpsurf[j],i,keepind))
            surf.append(tmpsurfsubfeat)
            surfla.append(surfla1)
        # surf dims: (featuresets, surfaces, prefeaturesets) with each one enclosing (samples, features)
        surf = np.array(surf).transpose()[:,:-1,:]
        # surfla dims: (samples, surfaces, prefeaturesets)
        surfla = np.array(surfla).transpose()
        if saveload:
            np.savez(surffile,surf=surf,surfla=surfla)
    if (printit):
        print surf.shape, surfla.shape
    return surf, surfla

############ PIPELINE OF TRANSFORMATIONS ############
def make_pipe_clf(scaler,feature_selection,decomp,clf):
    """returns a pipeline of inputs:
    -> scaler            : first normalize
    -> feature_selection : then perform feature selection
    -> decomp            : followed by PCA
    -> clf               : and finally the desired classifier
    <- pipeline          : output pipeline
    """
    pipeline = Pipeline([('scaler', scaler),
                         ('feature_selection', feature_selection),
                         ('decomp', decomp),
                         ('classifier', clf) ])
    return pipeline

############ TRAINING with 1 surface each time, out of 6 surfaces in total ##############
def filename1(i=0,j=0,k=0,l=0,retpath=0):
    """function for the filename of the selected combination for training per 1 surface
    -> i : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> k : surface id trained on
    -> l : surface id tested on
    <- filename
    """
    filepath = respath+'1/'
    ensure_dir(filepath)
    if retpath:
        return filepath
    else:
        return filepath+'fs_'+str(i)+'_subfs_'+str(j)+'_tr_'+str(k)+'_ts_'+str(l)+'.npz'

def cross_fit1(i,j,k,kmax,l,data,labels,data2,labels2,pipe,printit=True):
    """function for fitting model per 1 surface
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> k              : surface id trained on
    -> kmax           : maximum surfaces
    -> l              : surface id tested on
    -> data, labels   : training data and labels
    -> data2, labels2 : testing data and labels
    -> pipe           : the desired pipeline configuration
    <- no output, saved model and confusion matrix in corresponding filename.npz
    """
    fileid = filename1(i,j,k,l)
    if not os.path.isfile(fileid):
        if (printit):
            print i,j,k,l
        if k==l: # perform K-fold cross-validation
            folds = cv.split(data, labels)
            cm_all = np.zeros((2,2))
            for fold, (train_ind, test_ind) in enumerate(folds):
                x_train, x_test = data[train_ind], data[test_ind]
                y_train, y_test = labels[train_ind], labels[test_ind]
                model = pipe.fit(x_train,y_train)
                y_pred = model.predict(x_test)
                cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_all += cm/5.
            np.savez(fileid,cm=cm_all,model=np.array([model]))
        else: # perform cross-check
            tr_data = data
            tr_labels = labels
            ts_data = data2
            ts_labels = labels2
            # Check if model already existent, but not the cross-validated one (on the same surface)
            model = []
            for m in range(kmax):
                tmpcopyfileid = filename1(i,j,k,m)
                if k!=m and os.path.isfile(tmpcopyfileid):
                    if (printit):
                        print 'Found precomputed model of '+str(k)+', tested on '+str(m)+'. Testing on '+str(l)+'...'
                    model = np.load(tmpcopyfileid)['model'][0]
                    break
            if model==[]: # model not found precomputed
                if (printit):
                    print 'Fitting on '+str(k)+', testing on '+str(l)+'...'
                model = pipe.fit(tr_data,tr_labels)
            y_pred = model.predict(ts_data)
            cm = confusion_matrix(y_pred=y_pred, y_true=ts_labels)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.savez(fileid,cm=cm,model=np.array([model]))

def init_steps1(i,j,jmax,surf,surfla,printit=True):
    """function for helping parallelization of computations per 1 surface
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> jmax           : number of all subfeaturesets
    -> surf, surfla   : surface data and labels
    """
    if j==jmax:
        featsel = SelectKBest(k=1000,score_func= mutual_info_classif)
    else:
        featsel = SelectKBest(k='all',score_func= mutual_info_classif)
    pipe = make_pipe_clf(scaler, featsel, decomp, classifiers[2])
    for k in range(surf.shape[0]): # for every training surface
        for l in range(surf.shape[0]): # for every testing surface
            cross_fit1(i,j,k,surf.shape[0],l,surf[k],surfla[:,k],surf[l],surfla[:,l],pipe,printit)

def train_1_surface(surf,surfla,n=-1,printit=True):
    """Parallel training -on surface level- of all combinations on 1 surface
    -> n              : number of cores to run in parallel,
                        input of joblib's Parallel (n=-1 means all available cores)
    -> surf, surfla   : surface data and labels
    *** Cross surface validation, TRAINING with 1 surface each time, out of 6 surfaces in total
    total= 4 (featuresets) * [comb(6,1)*6] (surface combinations: trained on 1, tested on 1) * 1 (prefeatureset)
         = 4*6*6*1 = 144 different runs-files.
    Note that comb(n,r) = n!/(r!(n-r)!)
    """
    if (printit):
        print "-------------------------- TRAINING all combinations per 1 surface -----------------------------------"
    for i in range(len(prefeatid)-1):
        _ = [Parallel(n_jobs=n)([delayed(init_steps1) (i,j,surf.shape[0]-1,surf[j,:,i],surfla[:,:,i],printit)
                                  for j in range(surf.shape[0])])]

def bargraph_perf_gen1(maxsurf,printit=True):
    """Perf file for bargraph generation using bargraph tool, for 1 surface"""
    if (printit):
        print "---------------------------- Generating perf files for 1 surface -------------------------------------"
    prefeats = prefeatnames[prefeatid][:-1]
    # prefeatures, subfeatures, trained, tested, (TP,TN,FN,FP)
    acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,4))
    # prefeatures, subfeatures, trained, cross_val_self_accuracy, (TP,TN,FN,FP)
    self_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,1,4))
    # features, subfeatures, trained, (tested avg, tested std), (TP,TN,FN,FP)
    cross_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,2,4))
    initial_str = "# clustered and stacked graph bogus data\n=stackcluster;TP;TN;FN;FP\n"+\
                  "colors=med_blue,dark_green,yellow,red\n=nogridy\n=noupperright\nfontsz=5\nlegendx=right\n"+\
                  "legendy=center\ndatascale=50\nyformat=%g%%\nxlabel=TrainedON-TestedON\nylabel=Metrics\n=table"
    respath = filename1(retpath=1)
    for i in range(len(prefeats)):
        outname = respath+prefeats[i]
        outfile = outname+'.perf'
        outfile1 = outname+'_selfaccuracy.perf'
        outfile2 = outname+'_crossaccuracy.perf'
        out = open(outfile,'w+')
        out.write(initial_str+"\n")
        out1 = open(outfile1,'w+')
        out1.write(initial_str+"\n")
        out2 = open(outfile2,'w+')
        out2.write(initial_str+"\n")
        for k in range(maxsurf):
            for k2 in range(maxsurf):
                out.write("multimulti="+str(k)+"-"+str(k2)+"\n")
                for j in range(len(subfeats)):
                    fileid = filename1(i,j,k,k2)
                    tmp = np.load(fileid)['cm']
                    # print to outfile
                    acc[i,j,k,k2,0] = round(tmp[1,1],2) # TP
                    acc[i,j,k,k2,1] = round(tmp[0,0],2) # TN
                    acc[i,j,k,k2,2] = 1-round(tmp[1,1],2) # FN
                    acc[i,j,k,k2,3] = 1-round(tmp[0,0],2) # FP
                    out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],acc[i,j,k,k2,0],acc[i,j,k,k2,1],
                                                                        acc[i,j,k,k2,2],acc[i,j,k,k2,3]))
                    # prepare and print to outfile1
                    if k == k2:
                        if j == 0:
                            out1.write("multimulti="+str(k)+"-"+str(k2)+"\n")
                        self_acc[i,j,k,0,:] = acc[i,j,k,k2,:]
                        out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],self_acc[i,j,k,0,0],
                                                                 self_acc[i,j,k,0,1],self_acc[i,j,k,0,2],
                                                                 self_acc[i,j,k,0,3]))
                    # prepare and print to outfile2
                    if k != k2:
                        # all values of corresponding subfeatureset j have been filled to compute avg and std
                        if (k < maxsurf-1 and k2 == maxsurf-1) or (k == maxsurf-1 and k2 == maxsurf-2):
                            if j == 0:
                                out2.write("multimulti="+str(k)+"\n")
                            t = range(maxsurf)
                            t.remove(k)
                            cross_acc[i,j,k,0,:] = np.mean(acc[i,j,k,t,:], axis=0) # avg
                            # cross_acc[i,j,k,1,:] = np.std(acc[i,j,k,t,:], axis=0) # std
                            out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j], cross_acc[i,j,k,0,0],
                                                                     cross_acc[i,j,k,0,1], cross_acc[i,j,k,0,2],
                                                                     cross_acc[i,j,k,0,3]))
        out.write("multimulti=AVG\n")
        out1.write("multimulti=AVG\n")
        out2.write("multimulti=AVG\n")
        for j in range(4):
            avgacc = np.mean(np.mean(acc[i,j,:,:,:], axis=0), axis=0)
            out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j], avgacc[0], avgacc[1], avgacc[2], avgacc[3]))
            avgselfacc = np.mean(self_acc[i,j,:,0,:], axis=0)
            out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j], avgselfacc[0], avgselfacc[1],
                                                                  avgselfacc[2], avgselfacc[3]))
            avgcrossacc0 = np.mean(cross_acc[i,j,:,0,:], axis=0)
            # avgcrossacc1 = np.std(cross_acc[i,j,:,0,:], axis=0)
            out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j], avgcrossacc0[0], avgcrossacc0[1],
                                                                  avgcrossacc0[2], avgcrossacc0[3]))
        out.close()
        out1.close()
        out2.close()

############ TRAINING with 2 surfaces each time, out of 6 surfaces in total ##############
def filename2(i=0,j=0,k1=0,k2=0,l=0,retpath=0):
    """function for the filename of the selected combination for training per 2 surfaces
    -> i  : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j  : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> ki : surface ids trained on
    -> l  : surface id tested on
    <- filename
    """
    filepath = respath+'2/'
    ensure_dir(filepath)
    if retpath:
        return filepath
    else:
        return filepath+'fs_'+str(i)+'_subfs_'+str(j)+'_tr1_'+str(k1)+'_tr2_'+str(k2)+'_ts_'+str(l)+'.npz'

def cross_fit2(i,j,k1,k2,kmax,l,data,labels,data2,labels2,pipe,printit=True):
    """function for fitting model per 2 surfaces
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> ki             : surface ids trained on
    -> kmax           : maximum surfaces
    -> l              : surface id tested on
    -> data, labels   : training data and labels
    -> data2, labels2 : testing data and labels
    -> pipe           : the desired pipeline configuration
    <- no output, saved model and confusion matrix in corresponding filename.npz
    """
    fileid = filename2(i,j,k1,k2,l)
    if not os.path.isfile(fileid):
        if (printit):
            print i,j,k1,k2,l
        if k1==l or k2==l: # perform K-fold
            if (printit):
                print 'Fitting on '+str(k1)+"-"+str(k2)+', cross-validating on '+str(l)+'...'
            if l == k1: # copy if existent from the other sibling file
                tmpcopyfileid = filename2(i,j,k1,k2,k2)
            else:   # same as above
                tmpcopyfileid = filename2(i,j,k1,k2,k1)
            if not os.path.isfile(tmpcopyfileid):
                folds = cv.split(data, labels)
                cm_all = np.zeros((2,2))
                for fold, (train_ind, test_ind) in enumerate(folds):
                    x_train, x_test = data[train_ind], data[test_ind]
                    y_train, y_test = labels[train_ind], labels[test_ind]
                    model = pipe.fit(x_train,y_train)
                    y_pred = model.predict(x_test)
                    cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    cm_all += cm/5.
            else:
                cm_all = np.load(tmpcopyfileid)['cm']
                model = np.load(tmpcopyfileid)['model'][0]
            np.savez(fileid,cm=cm_all,model=np.array([model]))
        else: # perform cross-check
            tr_data = data
            tr_labels = labels
            ts_data = data2
            ts_labels = labels2
            model = []
            for m in range(kmax):
                tmpcopyfileid = filename2(i,j,k1,k2,m)
                if k1!=m and k2!=m and os.path.isfile(tmpcopyfileid):
                    if (printit):
                        print 'Found precomputed model of '+str(k1)+str(k2)+', tested on '+str(m)+'. Testing on '+str(l)+'...'
                    model = np.load(tmpcopyfileid)['model'][0]
                    break
            if model==[]: # model not found precomputed
                if (printit):
                    print 'Fitting on '+str(k1)+"-"+str(k2)+', testing on '+str(l)+'...'
                model = pipe.fit(tr_data,tr_labels)
            y_pred = model.predict(ts_data)
            cm = confusion_matrix(y_pred=y_pred, y_true=ts_labels)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.savez(fileid,cm=cm,model=np.array([model]))

def init_steps2(i,j,jmax,surf,surfla,printit=True):
    """function for helping parallelization of computations per 2 surfaces
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> jmax           : number of all subfeaturesets
    -> surf, surfla   : surface data and labels
    """
    if j==jmax:
        featsel = SelectKBest(k=1000,score_func= mutual_info_classif)
    else:
        featsel = SelectKBest(k='all',score_func= mutual_info_classif)
    pipe = make_pipe_clf(scaler, featsel, decomp, classifiers[2])
    for k1 in range(surf.shape[0]): # for every training surface1
        for k2 in range(surf.shape[0]): # for every training surface2
            if k2 > k1:
                for l in range(surf.shape[0]): # for every testing surface
                    tr_surf = np.concatenate((surf[k1],surf[k2]),axis=0)
                    tr_surfla = np.concatenate((surfla[:,k1],surfla[:,k2]),axis=0)
                    ts_surf, ts_surfla = surf[l], surfla[:,l]
                    cross_fit2(i,j,k1,k2,surf.shape[0],l,tr_surf,tr_surfla,ts_surf,ts_surfla,pipe,printit)

def train_2_surface(surf,surfla,n=-1,printit=True):
    """Parallel training -on surface level- of all combinations on 2 surfaces
    -> n              : number of cores to run in parallel,
                        input of joblib's Parallel (n=-1 means all available cores)
    -> surf, surfla   : surface data and labels
    *** Cross surface validation, TRAINING with 2 surfaces each time, out of 6 surfaces in total
    total= 4 (featuresets) * [comb(6,2)*6] (surface combinations: trained on 2, tested on 1) * 1 (prefeatureset)
         = 4*15*6*1 = 360 different runs-files.
    Note that comb(n,r) = n!/(r!(n-r)!)
    """
    if (printit):
        print "-------------------------- TRAINING all combinations per 2 surfaces ----------------------------------"
    for i in range(len(prefeatid)-1):
        _ = [Parallel(n_jobs=n)([delayed(init_steps2) (i,j,surf.shape[0]-1,surf[j,:,i],surfla[:,:,i],printit)
                                 for j in range(surf.shape[0])])]

def bargraph_perf_gen2(maxsurf,printit=True):
    """Perf file for bargraph generation using bargraph tool, for 2 surfaces"""
    if (printit):
        print "---------------------------- Generating perf files for 2 surfaces ------------------------------------"
    prefeats = prefeatnames[prefeatid][:-1]
    # prefeatures, subfeatures, trained, tested, (TP,TN,FN,FP)
    acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,4))
    # features, subfeatures, (TP,TN,FN,FP) -> avg over all tested surfaces
    avg = np.zeros((len(prefeats),len(subfeats),4))
    # prefeatures, subfeatures, trained, cross_val_self_accuracy, (TP,TN,FN,FP)
    self_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,1,4))
    # features, subfeatures, (TP,TN,FN,FP) -> avg over all self tested surfaces
    avgs = np.zeros((len(prefeats),len(subfeats),4))
    # features, subfeatures, trained, (tested avg, tested std), (TP,TN,FN,FP)
    cross_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,2,4))
     # features, subfeatures, (TP,TN,FN,FP) -> avg over all cross tested surfaces
    avgc = np.zeros((len(prefeats),len(subfeats),4))
    initial_str = "# clustered and stacked graph bogus data\n=stackcluster;TP;TN;FN;FP\n"+\
                  "colors=med_blue,dark_green,yellow,red\n=nogridy\n=noupperright\nfontsz=5\nlegendx=right\n"+\
                  "legendy=center\ndatascale=50\nyformat=%g%%\nxlabel=TrainedON-TestedON\nylabel=Metrics\n=table"
    respath = filename2(retpath=1)
    for i in range(len(prefeats)):
        outname = respath+prefeats[i]
        outfile = outname+'.perf'
        outfile1 = outname+'_selfaccuracy.perf'
        outfile2 = outname+'_crossaccuracy.perf'
        out = open(outfile,'w+')
        out.write(initial_str+"\n")
        out1 = open(outfile1,'w+')
        out1.write(initial_str+"\n")
        out2 = open(outfile2,'w+')
        out2.write(initial_str+"\n")
        for k1 in range(maxsurf):
            for k2 in range(maxsurf):
                if k2 > k1:
                    for l in range(maxsurf):
                        out.write("multimulti="+str(k1)+str(k2)+"-"+str(l)+"\n")
                        for j in range(len(subfeats)):
                            fileid = filename2(i,j,k1,k2,l)
                            tmp = np.load(fileid)['cm']
                            acc[i,j,k1,k2,l,0] = round(tmp[1,1],2) # TP
                            acc[i,j,k1,k2,l,1] = round(tmp[0,0],2) # TN
                            acc[i,j,k1,k2,l,2] = 1-round(tmp[1,1],2) # FN
                            acc[i,j,k1,k2,l,3] = 1-round(tmp[0,0],2) # FP
                            avg[i,j,:] += acc[i,j,k1,k2,l,:]
                            out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],acc[i,j,k1,k2,l,0],
                                                                    acc[i,j,k1,k2,l,1],acc[i,j,k1,k2,l,2],
                                                                    acc[i,j,k1,k2,l,3]))
                            if l == k1 or l == k2: # selc accuracy
                                if j == 0 and l == k2:
                                    out1.write("multimulti="+str(k1)+str(k2)+"-"+str(l)+"\n")
                                self_acc[i,j,k1,k2,0,:] = acc[i,j,k1,k2,l]
                                avgs[i,j,:] += self_acc[i,j,k1,k2,0,:]
                                if l == k2:
                                    out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                             self_acc[i,j,k1,k2,0,0],
                                                                             self_acc[i,j,k1,k2,0,1],
                                                                             self_acc[i,j,k1,k2,0,2],
                                                                             self_acc[i,j,k1,k2,0,3]))
                            if l != k1 and l != k2:
                                t = range(maxsurf)
                                t.remove(k1)
                                t.remove(k2)
                                if (l == t[-1]):
                                    if j == 0:
                                        out2.write("multimulti="+str(k1)+str(k2)+"\n")
                                    cross_acc[i,j,k1,k2,0,:] = np.mean(acc[i,j,k1,k2,t,:], axis=0) # avg
                                    # cross_acc[i,j,k1,k2,1,:] = np.std(acc[i,j,k1,k2,t,:], axis=0) # std
                                    avgc[i,j,:] += cross_acc[i,j,k1,k2,0,:]
                                    out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                             cross_acc[i,j,k1,k2,0,0],
                                                                             cross_acc[i,j,k1,k2,0,1],
                                                                             cross_acc[i,j,k1,k2,0,2],
                                                                             cross_acc[i,j,k1,k2,0,3]))
        out.write("multimulti=AVG\n")
        out1.write("multimulti=AVG\n")
        out2.write("multimulti=AVG\n")
        avg /= comb(maxsurf,2)*maxsurf*1.
        avgs /= comb(maxsurf,2)*2.
        avgc /= comb(maxsurf,2)*1.
        for j in range(len(subfeats)):
            out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avg[i,j,0],avg[i,j,1],avg[i,j,2],avg[i,j,3]))
            out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avgs[i,j,0],avgs[i,j,1],avgs[i,j,2],avgs[i,j,3]))
            out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avgc[i,j,0],avgc[i,j,1],avgc[i,j,2],avgc[i,j,3]))
        out.close()
        out1.close()
        out2.close()

############ TRAINING with 3 surfaces each time, out of 6 surfaces in total ##############
def filename3(i=0,j=0,k1=0,k2=0,k3=0,l=0,retpath=0):
    """function for the filename of the selected combination for training per 3 surfaces
    -> i  : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j  : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> ki : surface ids trained on
    -> l  : surface id tested on
    <- filename
    """
    filepath = respath+'3/'
    ensure_dir(filepath)
    if retpath:
        return filepath
    else:
        return filepath+'fs_'+str(i)+'_subfs_'+str(j)+'_tr1_'+str(k1)+'_tr2_'+str(k2)+'_tr3_'+str(k3)+'_ts_'+str(l)+'.npz'

def cross_fit3(i,j,k1,k2,k3,kmax,l,data,labels,data2,labels2,pipe,printit=True):
    """function for fitting model per 3 surfaces
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> ki             : surface ids trained on
    -> kmax           : maximum surfaces
    -> l              : surface id tested on
    -> data, labels   : training data and labels
    -> data2, labels2 : testing data and labels
    -> pipe           : the desired pipeline configuration
    <- no output, saved model and confusion matrix in corresponding filename.npz
    """
    fileid = filename3(i,j,k1,k2,k3,l)
    if not os.path.isfile(fileid):
        if (printit):
            print i,j,k1,k2,k3,l
        if k1==l or k2==l or k3==l: # perform K-fold
            if (printit):
                print 'Fitting on '+str(k1)+"-"+str(k2)+"-"+str(k3)+', cross-validating on '+str(l)+'...'
            if l == k1: # copy if existent from the other sibling file
                tmpcopyfileid1 = filename3(i,j,k1,k2,k3,k2)
                tmpcopyfileid2 = filename3(i,j,k1,k2,k3,k3)
            elif l == k2:   # same as above
                tmpcopyfileid1 = filename3(i,j,k1,k2,k3,k1)
                tmpcopyfileid2 = filename3(i,j,k1,k2,k3,k3)
            else:
                tmpcopyfileid1 = filename3(i,j,k1,k2,k3,k1)
                tmpcopyfileid2 = filename3(i,j,k1,k2,k3,k2)
            if not os.path.isfile(tmpcopyfileid1) and not os.path.isfile(tmpcopyfileid2):
                folds = cv.split(data, labels)
                cm_all = np.zeros((2,2))
                for fold, (train_ind, test_ind) in enumerate(folds):
                    x_train, x_test = data[train_ind], data[test_ind]
                    y_train, y_test = labels[train_ind], labels[test_ind]
                    model = pipe.fit(x_train,y_train)
                    y_pred = model.predict(x_test)
                    cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    cm_all += cm/5.
            else:
                if os.path.isfile(tmpcopyfileid1):
                    cm_all = np.load(tmpcopyfileid1)['cm']
                    model = np.load(tmpcopyfileid1)['model'][0]
                else:
                    cm_all = np.load(tmpcopyfileid2)['cm']
                    model = np.load(tmpcopyfileid2)['model'][0]
            np.savez(fileid,cm=cm_all,model=np.array([model]))
        else: # perform cross-check
            tr_data = data
            tr_labels = labels
            ts_data = data2
            ts_labels = labels2
            model = []
            for m in range(kmax):
                tmpcopyfileid = filename3(i,j,k1,k2,k3,m)
                if k1!=m and k2!=m and k3!=m and os.path.isfile(tmpcopyfileid):
                    if (printit):
                        print 'Found precomputed model of '+str(k1)+str(k2)+str(k3)+', tested on '+str(m)+'. Testing on '+str(l)+'...'
                    model = np.load(tmpcopyfileid)['model'][0]
                    break
            if model==[]: # model not found precomputed
                if (printit):
                    print 'Fitting on '+str(k1)+"-"+str(k2)+"-"+str(k3)+', testing on '+str(l)+'...'
                model = pipe.fit(tr_data,tr_labels)
            y_pred = model.predict(ts_data)
            cm = confusion_matrix(y_pred=y_pred, y_true=ts_labels)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.savez(fileid,cm=cm,model=np.array([model]))

def init_steps3(i,j,jmax,surf,surfla,printit=True):
    """function for helping parallelization of computations per 3 surfaces
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> jmax           : number of all subfeaturesets
    -> surf, surfla   : surface data and labels
    """
    if j==jmax:
        featsel = SelectKBest(k=1000,score_func= mutual_info_classif)
    else:
        featsel = SelectKBest(k='all',score_func= mutual_info_classif)
    pipe = make_pipe_clf(scaler, featsel, decomp, classifiers[2])
    for k1 in range(surf.shape[0]): # for every training surface1
        for k2 in range(surf.shape[0]): # for every training surface2
            if k2 > k1:
                for k3 in range(surf.shape[0]):
                    if k3 > k2:
                        for l in range(surf.shape[0]): # for every testing surface
                            tr_surf = np.concatenate((surf[k1],surf[k2],surf[k3]),axis=0)
                            tr_surfla = np.concatenate((surfla[:,k1],surfla[:,k2],surfla[:,k3]),axis=0)
                            ts_surf, ts_surfla = surf[l], surfla[:,l]
                            cross_fit3(i,j,k1,k2,k3,surf.shape[0],l,tr_surf,tr_surfla,ts_surf,ts_surfla,pipe,printit)

def train_3_surface(surf,surfla,n=-1,printit=True):
    """Parallel training -on surface level- of all combinations on 3 surfaces
    -> n              : number of cores to run in parallel,
                        input of joblib's Parallel (n=-1 means all available cores)
    -> surf, surfla   : surface data and labels
    *** Cross surface validation, TRAINING with 3 surfaces each time, out of 6 surfaces in total
    total= 4 (featuresets) * [comb(6,3)*6] (surface combinations: trained on 3, tested on 1) * 1 (prefeatureset)
         = 4*20*6*1 = 480 different runs-files.
    Note that comb(n,r) = n!/(r!(n-r)!)
    """
    if (printit):
        print "-------------------------- TRAINING all combinations per 3 surfaces ----------------------------------"
    for i in range(len(prefeatid)-1):
        _ = [Parallel(n_jobs=n)([delayed(init_steps3) (i,j,surf.shape[0]-1,surf[j,:,i],surfla[:,:,i],printit)
                                 for j in range(surf.shape[0])])]

def bargraph_perf_gen3(maxsurf,printit=True):
    """Perf file for bargraph generation using bargraph tool, for 3 surfaces"""
    if (printit):
        print "---------------------------- Generating perf files for 3 surfaces ------------------------------------"
    prefeats = prefeatnames[prefeatid][:-1]
    # prefeatures, subfeatures, trained, tested, (TP,TN,FN,FP)
    acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,maxsurf,4))
    # features, subfeatures, (TP,TN,FN,FP) -> avg over all tested surfaces
    avg = np.zeros((len(prefeats),len(subfeats),4))
    # prefeatures, subfeatures, trained, cross_val_self_accuracy, (TP,TN,FN,FP)
    self_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,1,4))
    # features, subfeatures, (TP,TN,FN,FP) -> avg over all self tested surfaces
    avgs = np.zeros((len(prefeats),len(subfeats),4))
    # features, subfeatures, trained, (tested avg, tested std), (TP,TN,FN,FP)
    cross_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,2,4))
     # features, subfeatures, (TP,TN,FN,FP) -> avg over all cross tested surfaces
    avgc = np.zeros((len(prefeats),len(subfeats),4))
    initial_str = "# clustered and stacked graph bogus data\n=stackcluster;TP;TN;FN;FP\n"+\
                  "colors=med_blue,dark_green,yellow,red\n=nogridy\n=noupperright\nfontsz=5\nlegendx=right\n"+\
                  "legendy=center\ndatascale=50\nyformat=%g%%\nxlabel=TrainedON-TestedON\nylabel=Metrics\n=table"
    respath = filename3(retpath=1)
    for i in range(len(prefeats)):
        outname = respath+prefeats[i]
        outfile = outname+'.perf'
        outfile1 = outname+'_selfaccuracy.perf'
        outfile2 = outname+'_crossaccuracy.perf'
        out = open(outfile,'w+')
        out.write(initial_str+"\n")
        out1 = open(outfile1,'w+')
        out1.write(initial_str+"\n")
        out2 = open(outfile2,'w+')
        out2.write(initial_str+"\n")
        for k1 in range(maxsurf):
            for k2 in range(maxsurf):
                if k2 > k1:
                    for k3 in range(maxsurf):
                        if k3 > k2:
                            for l in range(maxsurf):
                                out.write("multimulti="+str(k1)+str(k2)+str(k3)+"-"+str(l)+"\n")
                                for j in range(len(subfeats)):
                                    fileid = filename3(i,j,k1,k2,k3,l)
                                    tmp = np.load(fileid)['cm']
                                    acc[i,j,k1,k2,k3,l,0] = round(tmp[1,1],2) # TP
                                    acc[i,j,k1,k2,k3,l,1] = round(tmp[0,0],2) # TN
                                    acc[i,j,k1,k2,k3,l,2] = 1-round(tmp[1,1],2) # FN
                                    acc[i,j,k1,k2,k3,l,3] = 1-round(tmp[0,0],2) # FP
                                    avg[i,j,:] += acc[i,j,k1,k2,k3,l,:]
                                    out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],acc[i,j,k1,k2,k3,l,0],
                                                                            acc[i,j,k1,k2,k3,l,1],
                                                                            acc[i,j,k1,k2,k3,l,2],
                                                                            acc[i,j,k1,k2,k3,l,3]))
                                    if l == k1 or l == k2 or l == k3: # selc accuracy
                                        if j == 0 and l == k3:
                                            out1.write("multimulti="+str(k1)+str(k2)+str(k3)+"-"+str(l)+"\n")
                                        self_acc[i,j,k1,k2,k3,0,:] = acc[i,j,k1,k2,k3,l]
                                        avgs[i,j,:] += self_acc[i,j,k1,k2,k3,0,:]
                                        if l == k3:
                                            out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                                     self_acc[i,j,k1,k2,k3,0,0],
                                                                                     self_acc[i,j,k1,k2,k3,0,1],
                                                                                     self_acc[i,j,k1,k2,k3,0,2],
                                                                                     self_acc[i,j,k1,k2,k3,0,3]))
                                    if l != k1 and l != k2 and l != k3:
                                        t = range(maxsurf)
                                        t.remove(k1)
                                        t.remove(k2)
                                        t.remove(k3)
                                        if (l == t[-1]):
                                            if j == 0:
                                                out2.write("multimulti="+str(k1)+str(k2)+str(k3)+"\n")
                                            # avg
                                            cross_acc[i,j,k1,k2,k3,0,:] = np.mean(acc[i,j,k1,k2,k3,t,:], axis=0)
                                            # std
                                            # cross_acc[i,j,k1,k2,k3,1,:] = np.std(acc[i,j,k1,k2,k3,t,:], axis=0)
                                            avgc[i,j,:] += cross_acc[i,j,k1,k2,k3,0,:]
                                            out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                                     cross_acc[i,j,k1,k2,k3,0,0],
                                                                                     cross_acc[i,j,k1,k2,k3,0,1],
                                                                                     cross_acc[i,j,k1,k2,k3,0,2],
                                                                                     cross_acc[i,j,k1,k2,k3,0,3]))
        out.write("multimulti=AVG\n")
        out1.write("multimulti=AVG\n")
        out2.write("multimulti=AVG\n")
        avg /= comb(maxsurf,3)*maxsurf*1.
        avgs /= comb(maxsurf,3)*3.
        avgc /= comb(maxsurf,3)*1.
        for j in range(len(subfeats)):
            out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avg[i,j,0],avg[i,j,1],avg[i,j,2],avg[i,j,3]))
            out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avgs[i,j,0],avgs[i,j,1],avgs[i,j,2],avgs[i,j,3]))
            out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avgc[i,j,0],avgc[i,j,1],avgc[i,j,2],avgc[i,j,3]))
        out.close()
        out1.close()
        out2.close()

############ TRAINING with 4 surfaces each time, out of 6 surfaces in total ##############
def filename4(i=0,j=0,k1=0,k2=0,k3=0,k4=0,l=0,retpath=0):
    """function for the filename of the selected combination for training per 4 surfaces
    -> i  : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j  : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> ki : surface ids trained on
    -> l  : surface id tested on
    <- filename
    """
    filepath = respath+'4/'
    ensure_dir(filepath)
    if retpath:
        return filepath
    else:
        return filepath+'fs_'+str(i)+'_subfs_'+str(j)+'_tr1_'+str(k1)+'_tr2_'+str(k2)+'_tr3_'+str(k3)+'_tr4_'+str(k4)+'_ts_'+str(l)+'.npz'

def cross_fit4(i,j,k1,k2,k3,k4,kmax,l,data,labels,data2,labels2,pipe,printit=True):
    """function for fitting model per 4 surfaces
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> ki             : surface ids trained on
    -> kmax           : maximum surfaces
    -> l              : surface id tested on
    -> data, labels   : training data and labels
    -> data2, labels2 : testing data and labels
    -> pipe           : the desired pipeline configuration
    <- no output, saved model and confusion matrix in corresponding filename.npz
    """
    fileid = filename4(i,j,k1,k2,k3,k4,l)
    if not os.path.isfile(fileid):
        if (printit):
            print i,j,k1,k2,k3,k4,l
        if k1==l or k2==l or k3==l or k4==l: # perform K-fold
            if (printit):
                print 'Fitting on '+str(k1)+"-"+str(k2)+"-"+str(k3)+"-"+str(k4)+', cross-validating on '+str(l)+'...'
            if l == k1: # copy if existent from the other sibling file
                tmpcopyfileid1 = filename4(i,j,k1,k2,k3,k4,k2)
                tmpcopyfileid2 = filename4(i,j,k1,k2,k3,k4,k3)
                tmpcopyfileid3 = filename4(i,j,k1,k2,k3,k4,k4)
            elif l == k2:   # same as above
                tmpcopyfileid1 = filename4(i,j,k1,k2,k3,k4,k1)
                tmpcopyfileid2 = filename4(i,j,k1,k2,k3,k4,k3)
                tmpcopyfileid3 = filename4(i,j,k1,k2,k3,k4,k4)
            elif l == k3:   # same as above
                tmpcopyfileid1 = filename4(i,j,k1,k2,k3,k4,k1)
                tmpcopyfileid2 = filename4(i,j,k1,k2,k3,k4,k2)
                tmpcopyfileid3 = filename4(i,j,k1,k2,k3,k4,k4)
            else:
                tmpcopyfileid1 = filename4(i,j,k1,k2,k3,k4,k1)
                tmpcopyfileid2 = filename4(i,j,k1,k2,k3,k4,k2)
                tmpcopyfileid3 = filename4(i,j,k1,k2,k3,k4,k3)
            if not os.path.isfile(tmpcopyfileid1) and not os.path.isfile(tmpcopyfileid2) and not os.path.isfile(tmpcopyfileid3):
                folds = cv.split(data, labels)
                cm_all = np.zeros((2,2))
                for fold, (train_ind, test_ind) in enumerate(folds):
                    x_train, x_test = data[train_ind], data[test_ind]
                    y_train, y_test = labels[train_ind], labels[test_ind]
                    model = pipe.fit(x_train,y_train)
                    y_pred = model.predict(x_test)
                    cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    cm_all += cm/5.
            else:
                if os.path.isfile(tmpcopyfileid1):
                    cm_all = np.load(tmpcopyfileid1)['cm']
                    model = np.load(tmpcopyfileid1)['model'][0]
                elif os.path.isfile(tmpcopyfileid2):
                    cm_all = np.load(tmpcopyfileid2)['cm']
                    model = np.load(tmpcopyfileid2)['model'][0]
                elif os.path.isfile(tmpcopyfileid3):
                    cm_all = np.load(tmpcopyfileid3)['cm']
                    model = np.load(tmpcopyfileid3)['model'][0]
            np.savez(fileid,cm=cm_all,model=np.array([model]))
        else: # perform cross-check
            tr_data = data
            tr_labels = labels
            ts_data = data2
            ts_labels = labels2
            model = []
            for m in range(kmax):
                tmpcopyfileid = filename4(i,j,k1,k2,k3,k4,m)
                if k1!=m and k2!=m and k3!=m and k4!=m and os.path.isfile(tmpcopyfileid):
                    if (printit):
                        print 'Found precomputed model of '+str(k1)+str(k2)+str(k3)+str(k4)+', tested on '+str(m)+'. Testing on '+str(l)+'...'
                    model = np.load(tmpcopyfileid)['model'][0]
                    break
            if model==[]: # model not found precomputed
                if (printit):
                    print 'Fitting on '+str(k1)+"-"+str(k2)+"-"+str(k3)+"-"+str(k4)+', testing on '+str(l)+'...'
                model = pipe.fit(tr_data,tr_labels)
            y_pred = model.predict(ts_data)
            cm = confusion_matrix(y_pred=y_pred, y_true=ts_labels)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.savez(fileid,cm=cm,model=np.array([model]))

def init_steps4(i,j,jmax,surf,surfla,printit=True):
    """function for helping parallelization of computations per 4 surfaces
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> jmax           : number of all subfeaturesets
    -> surf, surfla   : surface data and labels
    """
    if j==jmax:
        featsel = SelectKBest(k=1000,score_func= mutual_info_classif)
    else:
        featsel = SelectKBest(k='all',score_func= mutual_info_classif)
    pipe = make_pipe_clf(scaler, featsel, decomp, classifiers[2])
    for k1 in range(surf.shape[0]): # for every training surface1
        for k2 in range(surf.shape[0]): # for every training surface2
            if k2 > k1:
                for k3 in range(surf.shape[0]):
                    if k3 > k2:
                        for k4 in range(surf.shape[0]):
                            if k4 > k3:
                                for l in range(surf.shape[0]): # for every testing surface
                                    tr_surf = np.concatenate((surf[k1],surf[k2],surf[k3]),axis=0)
                                    tr_surfla = np.concatenate((surfla[:,k1],surfla[:,k2],surfla[:,k3]),axis=0)
                                    ts_surf, ts_surfla = surf[l], surfla[:,l]
                                    cross_fit4(i,j,k1,k2,k3,k4,surf.shape[0],l,
                                               tr_surf,tr_surfla,ts_surf,ts_surfla,pipe,printit)

def train_4_surface(surf,surfla,n=-1,printit=True):
    """Parallel training -on surface level- of all combinations on 4 surfaces
    -> n              : number of cores to run in parallel,
                        input of joblib's Parallel (n=-1 means all available cores)
    -> surf, surfla   : surface data and labels
    *** Cross surface validation, TRAINING with 2 surfaces each time, out of 6 surfaces in total
    total= 4 (featuresets) * [comb(6,4)*6] (surface combinations: trained on 4, tested on 1) * 1 (prefeatureset)
         = 4*15*6*1 = 360 different runs-files.
    Note that comb(n,r) = n!/(r!(n-r)!)
    """
    if (printit):
        print "-------------------------- TRAINING all combinations per 4 surfaces ----------------------------------"
    for i in range(len(prefeatid)-1):
        _ = [Parallel(n_jobs=n)([delayed(init_steps4) (i,j,surf.shape[0]-1,surf[j,:,i],surfla[:,:,i],printit)
                                 for j in range(surf.shape[0])])]

def bargraph_perf_gen4(maxsurf,printit=True):
    """Perf file for bargraph generation using bargraph tool, for 4 surfaces"""
    if (printit):
        print "---------------------------- Generating perf files for 4 surfaces ------------------------------------"
    prefeats = prefeatnames[prefeatid][:-1]
    # prefeatures, subfeatures, trained, tested, (TP,TN,FN,FP)
    acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,maxsurf,maxsurf,4))
    # features, subfeatures, (TP,TN,FN,FP) -> avg over all tested surfaces
    avg = np.zeros((len(prefeats),len(subfeats),4))
    # prefeatures, subfeatures, trained, cross_val_self_accuracy, (TP,TN,FN,FP)
    self_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,maxsurf,1,4))
    # features, subfeatures, (TP,TN,FN,FP) -> avg over all self tested surfaces
    avgs = np.zeros((len(prefeats),len(subfeats),4))
    # features, subfeatures, trained, (tested avg, tested std), (TP,TN,FN,FP)
    cross_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,maxsurf,2,4))
     # features, subfeatures, (TP,TN,FN,FP) -> avg over all cross tested surfaces
    avgc = np.zeros((len(prefeats),len(subfeats),4))
    initial_str = "# clustered and stacked graph bogus data\n=stackcluster;TP;TN;FN;FP\n"+\
                  "colors=med_blue,dark_green,yellow,red\n=nogridy\n=noupperright\nfontsz=5\nlegendx=right\n"+\
                  "legendy=center\ndatascale=50\nyformat=%g%%\nxlabel=TrainedON-TestedON\nylabel=Metrics\n=table"
    respath = filename4(retpath=1)
    for i in range(len(prefeats)):
        outname = respath+prefeats[i]
        outfile = outname+'.perf'
        outfile1 = outname+'_selfaccuracy.perf'
        outfile2 = outname+'_crossaccuracy.perf'
        out = open(outfile,'w+')
        out.write(initial_str+"\n")
        out1 = open(outfile1,'w+')
        out1.write(initial_str+"\n")
        out2 = open(outfile2,'w+')
        out2.write(initial_str+"\n")
        for k1 in range(maxsurf):
            for k2 in range(maxsurf):
                if k2 > k1:
                    for k3 in range(maxsurf):
                        if k3 > k2:
                            for k4 in range(maxsurf):
                                if k4 > k3:
                                    for l in range(maxsurf):
                                        out.write("multimulti="+str(k1)+str(k2)+str(k3)+str(k4)+"-"+str(l)+"\n")
                                        for j in range(len(subfeats)):
                                            fileid = filename4(i,j,k1,k2,k3,k4,l)
                                            tmp = np.load(fileid)['cm']
                                            acc[i,j,k1,k2,k3,k4,l,0] = round(tmp[1,1],2) # TP
                                            acc[i,j,k1,k2,k3,k4,l,1] = round(tmp[0,0],2) # TN
                                            acc[i,j,k1,k2,k3,k4,l,2] = 1-round(tmp[1,1],2) # FN
                                            acc[i,j,k1,k2,k3,k4,l,3] = 1-round(tmp[0,0],2) # FP
                                            avg[i,j,:] += acc[i,j,k1,k2,k3,k4,l,:]
                                            out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                                    acc[i,j,k1,k2,k3,k4,l,0],
                                                                                    acc[i,j,k1,k2,k3,k4,l,1],
                                                                                    acc[i,j,k1,k2,k3,k4,l,2],
                                                                                    acc[i,j,k1,k2,k3,k4,l,3]))
                                            if l == k1 or l == k2 or l == k3 or l == k4: # selc accuracy
                                                if j == 0 and l == k4:
                                                    out1.write("multimulti="+str(k1)+str(k2)+str(k3)+str(k4)+"-"+str(l)+"\n")
                                                self_acc[i,j,k1,k2,k3,k4,0,:] = acc[i,j,k1,k2,k3,k4,l]
                                                avgs[i,j,:] += self_acc[i,j,k1,k2,k3,k4,0,:]
                                                if l == k4:
                                                    out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                  self_acc[i,j,k1,k2,k3,k4,0,0],
                                                                  self_acc[i,j,k1,k2,k3,k4,0,1],
                                                                  self_acc[i,j,k1,k2,k3,k4,0,2],
                                                                  self_acc[i,j,k1,k2,k3,k4,0,3]))
                                            if l != k1 and l != k2 and l != k3 and l!= k4:
                                                t = range(maxsurf)
                                                t.remove(k1)
                                                t.remove(k2)
                                                t.remove(k3)
                                                t.remove(k4)
                                                if (l == t[-1]):
                                                    if j == 0:
                                                        out2.write("multimulti="+str(k1)+str(k2)+str(k3)+str(k4)+"\n")
                                                    cross_acc[i,j,k1,k2,k3,k4,0,:] = np.mean(acc[i,j,k1,k2,k3,k4,t,:], axis=0)
                                                    avgc[i,j,:] += cross_acc[i,j,k1,k2,k3,k4,0,:]
                                                    out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                  cross_acc[i,j,k1,k2,k3,k4,0,0],
                                                                  cross_acc[i,j,k1,k2,k3,k4,0,1],
                                                                  cross_acc[i,j,k1,k2,k3,k4,0,2],
                                                                  cross_acc[i,j,k1,k2,k3,k4,0,3]))
        out.write("multimulti=AVG\n")
        out1.write("multimulti=AVG\n")
        out2.write("multimulti=AVG\n")
        avg /= comb(maxsurf,4)*maxsurf*1.
        avgs /= comb(maxsurf,4)*4.
        avgc /= comb(maxsurf,4)*1.
        for j in range(len(subfeats)):
            out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avg[i,j,0],avg[i,j,1],avg[i,j,2],avg[i,j,3]))
            out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avgs[i,j,0],avgs[i,j,1],avgs[i,j,2],avgs[i,j,3]))
            out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avgc[i,j,0],avgc[i,j,1],avgc[i,j,2],avgc[i,j,3]))
        out.close()
        out1.close()
        out2.close()

############ TRAINING with 5 surfaces each time, out of 6 surfaces in total ##############
def filename5(i=0,j=0,k1=0,k2=0,k3=0,k4=0,k5=0,l=0,retpath=0):
    """function for the filename of the selected combination for training per 5 surfaces
    -> i  : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j  : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> ki : surface ids trained on
    -> l  : surface id tested on
    <- filename
    """
    filepath = respath+'5/'
    ensure_dir(filepath)
    if retpath:
        return filepath
    else:
        return filepath+'fs_'+str(i)+'_subfs_'+str(j)+'_tr1_'+str(k1)+'_tr2_'+str(k2)+'_tr3_'+str(k3)+'_tr4_'+str(k4)+'_tr5_'+str(k5)+'_ts_'+str(l)+'.npz'

def cross_fit5(i,j,k1,k2,k3,k4,k5,kmax,l,data,labels,data2,labels2,pipe,printit=True):
    """function for fitting model per 5 surfaces
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> ki             : surface ids trained on
    -> kmax           : maximum surfaces
    -> l              : surface id tested on
    -> data, labels   : training data and labels
    -> data2, labels2 : testing data and labels
    -> pipe           : the desired pipeline configuration
    <- no output, saved model and confusion matrix in corresponding filename.npz
    """
    fileid = filename5(i,j,k1,k2,k3,k4,k5,l)
    if not os.path.isfile(fileid):
        if (printit):
            print i,j,k1,k2,k3,k4,k5,l
        if k1==l or k2==l or k3==l or k4==l or k5==l: # perform K-fold
            if (printit):
                print 'Fitting on '+str(k1)+"-"+str(k2)+"-"+str(k3)+"-"+str(k4)+"-"+str(k5)+', cross-validating on '+str(l)+'...'
            if l == k1: # copy if existent from the other sibling file
                tmpcopyfileid1 = filename5(i,j,k1,k2,k3,k4,k5,k2)
                tmpcopyfileid2 = filename5(i,j,k1,k2,k3,k4,k5,k3)
                tmpcopyfileid3 = filename5(i,j,k1,k2,k3,k4,k5,k4)
                tmpcopyfileid4 = filename5(i,j,k1,k2,k3,k4,k5,k5)
            elif l == k2:   # same as above
                tmpcopyfileid1 = filename5(i,j,k1,k2,k3,k4,k5,k1)
                tmpcopyfileid2 = filename5(i,j,k1,k2,k3,k4,k5,k3)
                tmpcopyfileid3 = filename5(i,j,k1,k2,k3,k4,k5,k4)
                tmpcopyfileid4 = filename5(i,j,k1,k2,k3,k4,k5,k5)
            elif l == k3:   # same as above
                tmpcopyfileid1 = filename5(i,j,k1,k2,k3,k4,k5,k1)
                tmpcopyfileid2 = filename5(i,j,k1,k2,k3,k4,k5,k2)
                tmpcopyfileid3 = filename5(i,j,k1,k2,k3,k4,k5,k4)
                tmpcopyfileid4 = filename5(i,j,k1,k2,k3,k4,k5,k5)
            elif l == k4:   # same as above
                tmpcopyfileid1 = filename5(i,j,k1,k2,k3,k4,k5,k1)
                tmpcopyfileid2 = filename5(i,j,k1,k2,k3,k4,k5,k2)
                tmpcopyfileid3 = filename5(i,j,k1,k2,k3,k4,k5,k3)
                tmpcopyfileid4 = filename5(i,j,k1,k2,k3,k4,k5,k5)
            else:
                tmpcopyfileid1 = filename5(i,j,k1,k2,k3,k4,k5,k1)
                tmpcopyfileid2 = filename5(i,j,k1,k2,k3,k4,k5,k2)
                tmpcopyfileid3 = filename5(i,j,k1,k2,k3,k4,k5,k3)
                tmpcopyfileid4 = filename5(i,j,k1,k2,k3,k4,k5,k4)
            if not os.path.isfile(tmpcopyfileid1) and not os.path.isfile(tmpcopyfileid2)\
               and not os.path.isfile(tmpcopyfileid3) and not os.path.isfile(tmpcopyfileid4):
                folds = cv.split(data, labels)
                cm_all = np.zeros((2,2))
                for fold, (train_ind, test_ind) in enumerate(folds):
                    x_train, x_test = data[train_ind], data[test_ind]
                    y_train, y_test = labels[train_ind], labels[test_ind]
                    model = pipe.fit(x_train,y_train)
                    y_pred = model.predict(x_test)
                    cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    cm_all += cm/5.
            else:
                if os.path.isfile(tmpcopyfileid1):
                    cm_all = np.load(tmpcopyfileid1)['cm']
                    model = np.load(tmpcopyfileid1)['model'][0]
                elif os.path.isfile(tmpcopyfileid2):
                    cm_all = np.load(tmpcopyfileid2)['cm']
                    model = np.load(tmpcopyfileid2)['model'][0]
                elif os.path.isfile(tmpcopyfileid3):
                    cm_all = np.load(tmpcopyfileid3)['cm']
                    model = np.load(tmpcopyfileid3)['model'][0]
                elif os.path.isfile(tmpcopyfileid4):
                    cm_all = np.load(tmpcopyfileid4)['cm']
                    model = np.load(tmpcopyfileid4)['model'][0]
            np.savez(fileid,cm=cm_all,model=np.array([model]))
        else: # perform cross-check
            tr_data = data
            tr_labels = labels
            ts_data = data2
            ts_labels = labels2
            model = []
            for m in range(kmax):
                tmpcopyfileid = filename5(i,j,k1,k2,k3,k4,k5,m)
                if k1!=m and k2!=m and k3!=m and k4!=m and k5!=m and os.path.isfile(tmpcopyfileid):
                    if (printit):
                        print 'Found precomputed model of '+str(k1)+str(k2)+str(k3)+str(k4)+str(k5)+', tested on '+str(m)+'. Testing on '+str(l)+'...'
                    model = np.load(tmpcopyfileid)['model'][0]
                    break
            if model==[]: # model not found precomputed
                if (printit):
                    print 'Fitting on '+str(k1)+"-"+str(k2)+"-"+str(k3)+"-"+str(k4)+"-"+str(k5)+', testing on '+str(l)+'...'
                model = pipe.fit(tr_data,tr_labels)
            y_pred = model.predict(ts_data)
            cm = confusion_matrix(y_pred=y_pred, y_true=ts_labels)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            np.savez(fileid,cm=cm,model=np.array([model]))

def init_steps5(i,j,jmax,surf,surfla,printit=True):
    """function for helping parallelization of computations per 5 surfaces
    -> i              : prefeature id, among all computed prefeatures (0: |f|, ... see prefeatid)
    -> j              : subfeatureset among all features (0: AFFT, 1: FREQ, 2: TIME, 3: ALL)
    -> jmax           : number of all subfeaturesets
    -> surf, surfla   : surface data and labels
    """
    if j==jmax:
        featsel = SelectKBest(k=1000,score_func= mutual_info_classif)
    else:
        featsel = SelectKBest(k='all',score_func= mutual_info_classif)
    pipe = make_pipe_clf(scaler, featsel, decomp, classifiers[2])
    for k1 in range(surf.shape[0]): # for every training surface1
        for k2 in range(surf.shape[0]): # for every training surface2
            if k2 > k1:
                for k3 in range(surf.shape[0]):
                    if k3 > k2:
                        for k4 in range(surf.shape[0]):
                            if k4 > k3:
                                for k5 in range(surf.shape[0]):
                                    if k5 > k4:
                                        for l in range(surf.shape[0]): # for every testing surface
                                            tr_surf = np.concatenate((surf[k1],surf[k2],surf[k3]),axis=0)
                                            tr_surfla = np.concatenate((surfla[:,k1],surfla[:,k2],
                                                                        surfla[:,k3]),axis=0)
                                            ts_surf, ts_surfla = surf[l], surfla[:,l]
                                            cross_fit5(i,j,k1,k2,k3,k4,k5,surf.shape[0],l,
                                                       tr_surf,tr_surfla,ts_surf,ts_surfla,pipe,printit)

def train_5_surface(surf,surfla,n=-1,printit=True):
    """Parallel training -on surface level- of all combinations on 5 surfaces
    -> n              : number of cores to run in parallel,
                        input of joblib's Parallel (n=-1 means all available cores)
    -> surf, surfla   : surface data and labels
    *** Cross surface validation, TRAINING with 5 surfaces each time, out of 6 surfaces in total
    total= 4 (featuresets) * [comb(6,5)*6] (surface combinations: trained on 5, tested on 1) * 1 (prefeatureset)
         = 4*6*6*1 = 144 different runs-files.
    Note that comb(n,r) = n!/(r!(n-r)!)
    """
    if (printit):
        print "-------------------------- TRAINING all combinations per 5 surfaces ----------------------------------"
    for i in range(len(prefeatid)-1):
        _ = [Parallel(n_jobs=n)([delayed(init_steps5) (i,j,surf.shape[0]-1,surf[j,:,i],surfla[:,:,i],printit)
                                 for j in range(surf.shape[0])])]

def bargraph_perf_gen5(maxsurf,printit=True):
    """Perf file for bargraph generation using bargraph tool, for 5 surfaces"""
    if (printit):
        print "---------------------------- Generating perf files for 5 surfaces ------------------------------------"
    prefeats = prefeatnames[prefeatid][:-1]
    # prefeatures, subfeatures, trained, tested, (TP,TN,FN,FP)
    acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,maxsurf,maxsurf,maxsurf,4))
    # features, subfeatures, (TP,TN,FN,FP) -> avg over all tested surfaces
    avg = np.zeros((len(prefeats),len(subfeats),4))
    # prefeatures, subfeatures, trained, cross_val_self_accuracy, (TP,TN,FN,FP)
    self_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,maxsurf,maxsurf,1,4))
    # features, subfeatures, (TP,TN,FN,FP) -> avg over all self tested surfaces
    avgs = np.zeros((len(prefeats),len(subfeats),4))
    # features, subfeatures, trained, (tested avg, tested std), (TP,TN,FN,FP)
    cross_acc = np.zeros((len(prefeats),len(subfeats),maxsurf,maxsurf,maxsurf,maxsurf,maxsurf,2,4))
     # features, subfeatures, (TP,TN,FN,FP) -> avg over all cross tested surfaces
    avgc = np.zeros((len(prefeats),len(subfeats),4))
    initial_str = "# clustered and stacked graph bogus data\n=stackcluster;TP;TN;FN;FP\n"+\
                  "colors=med_blue,dark_green,yellow,red\n=nogridy\n=noupperright\nfontsz=5\nlegendx=right\n"+\
                  "legendy=center\ndatascale=50\nyformat=%g%%\nxlabel=TrainedON-TestedON\nylabel=Metrics\n=table"
    respath = filename5(retpath=1)
    for i in range(len(prefeats)):
        outname = respath+prefeats[i]
        outfile = outname+'.perf'
        outfile1 = outname+'_selfaccuracy.perf'
        outfile2 = outname+'_crossaccuracy.perf'
        out = open(outfile,'w+')
        out.write(initial_str+"\n")
        out1 = open(outfile1,'w+')
        out1.write(initial_str+"\n")
        out2 = open(outfile2,'w+')
        out2.write(initial_str+"\n")
        for k1 in range(maxsurf):
            for k2 in range(maxsurf):
                if k2 > k1:
                    for k3 in range(maxsurf):
                        if k3 > k2:
                            for k4 in range(maxsurf):
                                if k4 > k3:
                                    for k5 in range(maxsurf):
                                        if k5 > k4:
                                            for l in range(maxsurf):
                                                out.write("multimulti="+str(k1)+str(k2)+str(k3)+str(k4)
                                                                       +str(k5)+"-"+str(l)+"\n")
                                                for j in range(len(subfeats)):
                                                    fileid = filename5(i,j,k1,k2,k3,k4,k5,l)
                                                    tmp = np.load(fileid)['cm']
                                                    acc[i,j,k1,k2,k3,k4,k5,l,0] = round(tmp[1,1],2) # TP
                                                    acc[i,j,k1,k2,k3,k4,k5,l,1] = round(tmp[0,0],2) # TN
                                                    acc[i,j,k1,k2,k3,k4,k5,l,2] = 1-round(tmp[1,1],2) # FN
                                                    acc[i,j,k1,k2,k3,k4,k5,l,3] = 1-round(tmp[0,0],2) # FP
                                                    avg[i,j,:] += acc[i,j,k1,k2,k3,k4,k5,l,:]
                                                    out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                 acc[i,j,k1,k2,k3,k4,k5,l,0],
                                                                 acc[i,j,k1,k2,k3,k4,k5,l,1],
                                                                 acc[i,j,k1,k2,k3,k4,k5,l,2],
                                                                 acc[i,j,k1,k2,k3,k4,k5,l,3]))
                                                    # selc accuracy
                                                    if l == k1 or l == k2 or l == k3 or l == k4 or l == k5:
                                                        if j == 0 and l == k5:
                                                            out1.write("multimulti="+str(k1)+str(k2)
                                                                       +str(k3)+str(k4)+str(k5)+"-"+str(l)+"\n")
                                                        self_acc[i,j,k1,k2,k3,k4,k5,0,:] = acc[i,j,k1,k2,k3,k4,k5,l]
                                                        avgs[i,j,:] += self_acc[i,j,k1,k2,k3,k4,k5,0,:]
                                                        if l == k5:
                                                            out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                          self_acc[i,j,k1,k2,k3,k4,k5,0,0],
                                                                          self_acc[i,j,k1,k2,k3,k4,k5,0,1],
                                                                          self_acc[i,j,k1,k2,k3,k4,k5,0,2],
                                                                          self_acc[i,j,k1,k2,k3,k4,k5,0,3]))
                                                    if l != k1 and l != k2 and l != k3 and l!= k4 and l!= k5:
                                                        t = range(maxsurf)
                                                        t.remove(k1)
                                                        t.remove(k2)
                                                        t.remove(k3)
                                                        t.remove(k4)
                                                        t.remove(k5)
                                                        if (l == t[-1]):
                                                            if j == 0:
                                                                out2.write("multimulti="+str(k1)+str(k2)+str(k3)+str(k4)+str(k5)+"\n")
                                                            cross_acc[i,j,k1,k2,k3,k4,k5,0,:] = np.mean(acc[i,j,k1,k2,k3,k4,k5,t,:], axis=0)
                                                            avgc[i,j,:] += cross_acc[i,j,k1,k2,k3,k4,k5,0,:]
                                                            out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],
                                                                          cross_acc[i,j,k1,k2,k3,k4,k5,0,0],
                                                                          cross_acc[i,j,k1,k2,k3,k4,k5,0,1],
                                                                          cross_acc[i,j,k1,k2,k3,k4,k5,0,2],
                                                                          cross_acc[i,j,k1,k2,k3,k4,k5,0,3]))
        out.write("multimulti=AVG\n")
        out1.write("multimulti=AVG\n")
        out2.write("multimulti=AVG\n")
        avg /= comb(maxsurf,5)*maxsurf*1.
        avgs /= comb(maxsurf,5)*5.
        avgc /= comb(maxsurf,5)*1.
        for j in range(len(subfeats)):
            out.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avg[i,j,0],avg[i,j,1],avg[i,j,2],avg[i,j,3]))
            out1.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avgs[i,j,0],avgs[i,j,1],avgs[i,j,2],avgs[i,j,3]))
            out2.write("%s %.2f %.2f %.2f %.2f\n" % (subfeats[j],avgc[i,j,0],avgc[i,j,1],avgc[i,j,2],avgc[i,j,3]))
        out.close()
        out1.close()
        out2.close()

def make_bargraphs_from_perf(i,maxsurf=6,printit=True):
    """Bargraph generation using bargraph tool, for i surfaces"""
    if (printit):
        print "---------------------------- Generating bar graphs for "+str(i+1)+" surfaces ------------------------------------"
    resfold = respath+str(i+1)+'/'
    allperf = glob.glob(resfold+"*.perf")
    maxperf = len(allperf)
    for k in range(len(allperf)):
        j = allperf[k]
        tmppdf = j[:-4]+"pdf"
        tmppng = j[:-4]+"png"
        with open(tmppdf, "w") as f1:
            call([tool,"-pdf",j],stdout=f1)
        with open(tmppng, "w") as f2:
            call([tool,"-png","-non-transparent",j],stdout=f2)
        img = mpimg.imread(tmppng)
        if k!=0:
            plt.subplot(maxsurf,maxperf-1,k+i*(maxperf-1))
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])

############ PREDICTING ACCURACY FOR ATI SENSOR DATA ##############
def testing_accuracy_simple(surf, surfla, Xsp, Ysp, keepind=[-1], printit=True, ltest=6):
    if len(keepind) == 0 or keepind[0] == -1:
        keepind = range(len(featnames))
    fileid = filename1(0,3,0,5)            #  all  features, 1 trained surface(surf 0)
    fileidb = filename1(0,0,0,5)           # |FFT| features, 1 trained surface(surf 0)
    # fileid5 = filename5(0,3,0,1,2,3,4,5)   #  all  features, 5 trained surfaces(surf 0-4)
    # fileid5b = filename5(0,0,0,1,2,3,4,5)  # |FFT| features, 5 trained surfaces(surf 0-4)
    model = np.load(fileid)['model'][0]
    modelb = np.load(fileidb)['model'][0]
    # model5 = np.load(fileid5)['model'][0]
    # model5b = np.load(fileid5b)['model'][0]
    for i in range(ltest):
        Yout = model.predict(surf[3, i, 0])
        Youtb = modelb.predict(surf[0, i, 0])
        # Yout5 = model5.predict(surf[3, i, 0])
        # Yout5b = model5b.predict(surf[0, i, 0])
        if printit:
            # print i, Yout.shape, Youtb.shape#, Yout5.shape, Yout5b.shape
            pass
        Ysc = model.score(surf[3, i ,0], surfla[:, i, 0])
        Yscb = modelb.score(surf[0, i, 0], surfla[:, i, 0])
        # Ysc5 = model5.score(surf[3, i, 0], surfla[:, i, 0])
        # Ysc5b = model5b.score(surf[0, i ,0], surfla[:, i, 0])
        Ycm = confusion_matrix(y_pred=Yout, y_true=surfla[:, i, 0])
        Ycm = Ycm.astype('float') / Ycm.sum(axis=1)[:, np.newaxis]
        Ycmb = confusion_matrix(y_pred=Youtb, y_true=surfla[:, i, 0])
        Ycmb = Ycmb.astype('float') / Ycmb.sum(axis=1)[:, np.newaxis]
        # Ycm5 = confusion_matrix(y_pred=Yout5, y_true=surfla[:, i, 0])
        # Ycm5b = confusion_matrix(y_pred=Yout5b, y_true=surfla[:, i, 0])
        if printit:
            print "Accuracy for surface ", i, Ysc, Yscb #, Ysc5, Ysc5b
            print "TN(stable) and TP(slip) for surface ", i, Ycm[0,0], Ycm[1,1],'|', Ycmb[0,0], Ycmb[1,1]
    Youtn = model.predict(Xsp[2][:,keepind])
    Youtbn = modelb.predict(Xsp[2][:,-window-2:-window/2-1])
    # Yout5n = model5.predict(Xsp[2])
    # Yout5bn = model5b.predict(Xsp[2][:,-window-2:-window/2-1])
    Yscn = model.score(Xsp[2][:,keepind],Ysp[2])
    Yscbn = modelb.score(Xsp[2][:,-window-2:-window/2-1],Ysp[2])
    # Ysc5n = model5.score(Xsp[2],Ysp[2])
    # Ysc5bn = model5b.score(Xsp[2][:,-window-2:-window/2-1],Ysp[2])
    Ycmn = confusion_matrix(y_pred=Youtn, y_true=Ysp[2])
    Ycmn = Ycmn.astype('float') / Ycmn.sum(axis=1)[:, np.newaxis]
    Ycmbn = confusion_matrix(y_pred=Youtbn, y_true=Ysp[2])
    Ycmbn = Ycmbn.astype('float') / Ycmbn.sum(axis=1)[:, np.newaxis]
    # Ycm5n = confusion_matrix(y_pred=Yout5n, y_true=Ysp[2])
    # Ycm5bn = confusion_matrix(y_pred=Yout5bn, y_true=Ysp[2])
    print "======================================================================================"
    print "Accuracy for dataset   ", Yscn, Yscbn  #, Ysc5n, Ysc5bn
    print "TN(stable) and TP(slip) for dataset ", Ycmn[0,0], Ycmn[1,1],'|', Ycmbn[0,0], Ycmbn[1,1]
    print "======================================================================================"

############ PREDICTING ACCURACY FOR ATI SENSOR DATA DETAILED ##############
def testing_accuracy(surf, surfla, trsurf=[1, 5], ltest=6, printit=True):
    lsurf = len(trsurf)
    lsubfs = surf.shape[0]
    acc = np.zeros((lsurf,lsubfs,ltest,2))
    for r in range(lsurf):  # for each number of surfaces used for training
        for k in range(lsubfs):  # for each subfs
            filenames = glob.glob(respath + str(trsurf[r]) + "/fs_" + str(0) + "_subfs_" + str(k) + "_*.npz")
            numf = len(filenames)
            for i in range(ltest):  # for each testing surface
                curracc = np.zeros(numf)
                for n in range(numf):
                    model = np.load(filenames[n])['model'][0]
                    Ysc = model.score(surf[k, i ,0], surfla[:, i, 0])
                    curracc[n] = Ysc
                    # if printit:
                    #     print "Surf: ",trsurf[r],"subfs: ",k,"test_surf: ",i,"model: ",n,"Acc: ",Ysc
                acc[r,k,i,0] = np.mean(curracc)
                acc[r,k,i,1] = np.std(curracc)
                if printit:
                    print "Surf: ",trsurf[r],"subfs: ",k,"test_surf: ",i,"Acc_mean-std: ",acc[r,k,i,0], acc[r,k,i,1]
            if printit:
                print "Surf: ",trsurf[r],"subfs: ",k,"Acc_mean-std: ",np.mean(acc[r,k,:,0]), np.mean(acc[r,k,:,1])
    return acc

############ VISUALIZING ONLINE TESTING PROCEDURE ##############
def visualize(f, surf, surfla, chosensurf=5, plotpoints=200, save=False, printit=True):
    matplotlib.rcParams['text.usetex'] = True
    offset = window
    inp = f[chosensurf][offset-600:,:3]
    lab = f[chosensurf][offset-600:,-1]
    INP1 = surf[3,chosensurf,0]
    INP2 = surf[0,chosensurf,0]
    OUT = surfla[:,chosensurf,0]
    answerfreq = 5. # Hz
    # plotpoints = 200.                             # 200 datapoints answer for visual purposes
    # plotpoints = INP1.shape[0]*1.                 # like 1ms answers
    # plotpoints = answerfreq*inp.shape[0]/window   # like real time answers (200ms or 5Hz)
    skipINP = int(round(INP1.shape[0]/plotpoints))
    endsetINP = range(INP1.shape[0])[::skipINP]
    minlen = len(endsetINP)
    mult = (inp.shape[0]-offset)/INP1.shape[0]*1.
    tx = np.array(endsetINP[:minlen][:-1])*mult
    tfx = range(inp.shape[0])[:int(endsetINP[-1]*mult)]
    tfind = (np.array(tfx) + offset).tolist()
    if printit:
        print skipINP, len(endsetINP)
    endsetINP = endsetINP[:minlen][-1]
    if printit:
        print skipINP, endsetINP
    fileid = filename1(0,3,0,5)
    fileidb = filename1(0,0,0,5)
    model = np.load(fileid)['model'][0]
    modelb = np.load(fileidb)['model'][0]
    Yout = model.predict(INP1)
    Youtb = modelb.predict(INP2)
    if printit:
        print Yout.shape, Youtb.shape
    plt.rc('text', usetex=True)
    plt.rc('axes', linewidth=2)
    plt.rc('font', weight='bold')
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
    ax = plt.figure(figsize=(20,10))
    tf = np.linalg.norm(inp[tfind],axis=1)
    tl = lab[tfind]
    ty = Yout[:endsetINP:skipINP]+0.02
    tyb = Youtb[:endsetINP:skipINP]+0.04
    tyl = OUT[:endsetINP:skipINP]+0.06
    if printit:
        print tf.shape, ty.shape, len(tx)
    p1, = plt.plot(tfx,tf/max(tf),linewidth=5)
    plt.hold
    pl, = plt.plot(tfx,tl,linewidth=5,color='green')
    p = plt.scatter(tx,ty,color='red',s=30)
    pb = plt.scatter(tx,tyb,color='magenta',s=30)
    pbl = plt.scatter(tx,tyl,color='brown',s=30)
    plt.text(100, 0.15, r'\textbf{Stable}', ha="center", va="center", rotation=0,
                size=25)
    plt.text(100, 0.85, r'\textbf{Slip}', ha="center", va="center", rotation=0,
                size=25)
    plt.annotate('', fontsize=10, xy=(100, 0.05), xytext=(100, 0.12),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('', xy=(100, 0.98), xytext=(100, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel(r't ($1e^{-2} sec$)',fontsize=35)
    # plt.yticks([])
    plt.legend([p1,pl,p,pb,pbl],[r'$|\textbf{f}|$',r'$\textbf{Flabel}$',r'\textbf{outBOTH}',r'\textbf{outFFT}',r'$\textbf{TRlabel}$'], prop={'size': 35})
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    if save:
        savefig(datapath+'validation_ati.pdf', bbox_inches='tight')

###### Prediction function for new datasets
def prediction(dataset,keepind=[-1],k=1,n=6,scale=1.0,printit=False,plotit=False):
    if len(keepind) == 0 or keepind[0] == -1:
        keepind = range(len(featnames))
    print "Filename for prediction: "+dataset
    if dataset[-4:] == '.mat':
        atifile = datapath+dataset
        atifeatname = dataset[:-4]+'_'+featname+'_'+str(scale)+'_'
        atifeatfile = featpath+atifeatname+'.npz'
        atisurffile = featpath+atifeatname+'_'+str(len(keepind))+'_'+str(k)+'fing_'+str(n)+'surf.npz'
        atiXYfile = featpath+atifeatname+'_XY.npz'
        atiXYsplitfile = featpath+atifeatname+'_XYsplit.npz'
        f,l,fd,member,m1,m2 = data_prep(atifile,k=k,printit=printit)
        print np.max(f[0][:,:-1])
        for i in range(len(f)):
            f[i][:,:-1] = scale * f[i][:,:-1]
        print np.max(f[0][:,:-1])
        prefeat = compute_prefeat(f,printit)
        features, labels = feature_extraction(prefeat, member, atifeatfile, atifeatname,printit)
        new_labels = label_cleaning(prefeat,labels,member,printit=printit)
        X,Y,Yn,Xsp,Ysp = computeXY(features,labels,new_labels,m1,m2,atiXYfile,atiXYsplitfile,printit)
        surf, surfla = computeXY_persurf(Xsp,Ysp,atisurffile,keepind,n=n,k=k,printit=printit)
        ############ PREDICTING SCORE FOR ATI SENSOR DATA ROTATIONAL ##############
        testing_accuracy_simple(surf, surfla, Xsp, Ysp, keepind, ltest=n)
        ############ PREDICTING SCORE FOR ATI SENSOR DATA DETAILED ##############
        # _ = testing_accuracy(surf, surfla, ltest=6)
        surfnosplit, surflanosplit = computeXY_persurf(X,Y,atisurffile,keepind,n=n,k=k,saveload=False,printit=printit)
        for chosensurf in range(5):
            if plotit:
                visualize(f, surfnosplit, surflanosplit, chosensurf, plotpoints=200, printit=printit)
    else:
        print "Your dataset should be .mat file. You provided instead a file."+dataset[-3:]
