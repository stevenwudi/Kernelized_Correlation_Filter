"""
We import HDT result from matlab for python port
"""
import scipy.io as sio
import os
import json
import glob
import h5py
import numpy as np

seqs = os.listdir('.././results/DSST')
for seq in seqs:
    #f = h5py.File('.././results/DSST/'+seq)
    f = sio.loadmat('.././results/DSST/'+seq)
    dict = {}
    dict['seqName'] = seq[4:-4].replace('_', '-')
    dict['tracker'] = "DSST"
    dict['fps'] = f['results'][0][0][0][0][0][0][0][2][0][0].tolist()
    dict['startFrame'] = f['results'][0][0][0][0][0][0][0][4][0][0].tolist()
    dict['endFrame'] = f['results'][0][0][0][0][0][0][0][3][0][0].tolist()
    dict['evalType'] = "OPE"
    dict['resType'] = "rect"
    dict['res'] = f['results'][0][0][0][0][0][0][0][1].tolist()
    with open('.././results_TB100/OPE/DSST/'+ dict['seqName'].lower() + '.json', 'w') as outfile:
        json.dump(dict, outfile)
