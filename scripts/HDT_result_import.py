"""
We import HDT result from matlab for python port
"""
import scipy.io as sio
import os
import json
import glob
import h5py

if False:
    seqs = os.listdir('results/HDT_cvpr2016')
    for seq in seqs:
        mat =sio.loadmat('results/HDT_cvpr2016/'+seq)
        dict = {}
        dict['seqName'] = seq[:-8]
        dict['tracker'] = "HDT"
        dict['fps'] = mat['results'][0][0][0]['fps'][0][0][0].tolist()
        dict['startFrame'] = mat['results'][0][0][0]['startFrame'][0][0][0].tolist()
        dict['endFrame'] = mat['results'][0][0][0]['len'][0][0][0].tolist()
        dict['evalType'] = "OPE"
        dict['resType'] = "rect"
        dict['res'] = mat['results'][0][0][0]['res'][0].tolist()
        with open('results/OPE/HDT_cvpr2016/'+ dict['seqName'] + '.json', 'w') as outfile:
            json.dump(dict, outfile)

if False:

    seqs = glob.glob('results/C-COT_OTB_2015_results/OPE*')
    for seq in seqs:
        f_load = h5py.File(seq, "r")
        mat = f_load['results']
        dict = {}
        dict['seqName'] = seq[:-8]
        dict['tracker'] = "HDT"
        dict['fps'] = mat['results'][0][0][0]['fps'][0][0][0].tolist()
        dict['startFrame'] = mat['results'][0][0][0]['startFrame'][0][0][0].tolist()
        dict['endFrame'] = mat['results'][0][0][0]['len'][0][0][0].tolist()
        dict['evalType'] = "OPE"
        dict['resType'] = "rect"
        dict['res'] = mat['results'][0][0][0]['res'][0].tolist()
        with open('results/OPE/HDT_cvpr2016/'+ dict['seqName'] + '.json', 'w') as outfile:
            json.dump(dict, outfile)

seqs = os.listdir('./../results/HDT-OTB100/HDT-OPE100')
for seq in seqs:
    mat = sio.loadmat('./../results/HDT-OTB100/HDT-OPE100/' + seq)
    dict = {}
    dict['seqName'] = seq[:-8]
    dict['tracker'] = "HDT"
    dict['fps'] = mat['results'][0][0][0]['fps'][0][0][0].tolist()
    dict['startFrame'] = mat['results'][0][0][0]['startFrame'][0][0][0].tolist()
    dict['endFrame'] = mat['results'][0][0][0]['len'][0][0][0].tolist()
    dict['evalType'] = "OPE"
    dict['resType'] = "rect"
    dict['res'] = mat['results'][0][0][0]['res'][0].tolist()
    with open('./../results_TB100/OPE/HDT_cvpr2016/' + dict['seqName'].lower().replace('-', '_') + '.json', 'w') as outfile:
        json.dump(dict, outfile)

