"""
We import HDT result from matlab for python port
"""
import scipy.io as sio
import os
import json
from load_mat_file import loadmat
OBT_result_dir = '/home/stevenwudi/Documents/MATLAB/HDT-release/HDT/OBT100_result'

Tracker_names = ['HDT_code']


seqs = sorted(os.listdir(OBT_result_dir))
for seq in seqs:
    print(seq)
    for tracker_name in Tracker_names:
        idx = seq.find(tracker_name)
        tracker_name = tracker_name
        mat = loadmat(os.path.join(OBT_result_dir, seq))
        dict = {}
        dict['seqName'] = seq[:-4]
        dict['tracker'] = tracker_name
        dict['evalType'] = "OPE"
        dict['resType'] = "rect"
        if mat['res']['res'].shape[1] == 4:
            dict['res'] = mat['res']['res'].tolist()
            if not os.path.exists(os.path.join('./../results_TB100/OPE', tracker_name)):
                os.mkdir(os.path.join('./../results_TB100/OPE', tracker_name))
            with open(os.path.join('./../results_TB100/OPE', tracker_name, dict['seqName'].lower() + '.json'), 'w') as outfile:
                json.dump(dict, outfile)

