"""
We import HDT result from matlab for python port
"""
import scipy.io as sio
import os
import json
from load_mat_file import loadmat
UAV_result_dir = '/home/stevenwudi/Documents/Python_Project/UAV/results_OPE_OTB100'

Tracker_names = ['ASLA', 'CSK', 'DSST', 'IVT', 'KCF_GaussHog', 'KCF_LinearGray', 'KCF_LinearHog', 'MEEM', 'MUSTER',
                  'OAB', 'SAMF', 'SRDCF', 'Struck', 'TLD']


seqs = sorted(os.listdir(UAV_result_dir))
for seq in seqs:
    print(seq)
    for tracker_name in Tracker_names:
        idx = seq.find(tracker_name)
        if idx>-1:
            tracker_name = seq[idx:-4]
            mat = loadmat(os.path.join(UAV_result_dir, seq))
            dict = {}
            dict['seqName'] = seq[:idx-1].lower().replace('.', '-').replace('_', '-')
            dict['tracker'] = tracker_name
            dict['evalType'] = "OPE"
            dict['resType'] = "rect"
            if mat['results']['res'].shape[1] == 4:
                dict['res'] = mat['results']['res'].tolist()
                if not os.path.exists(os.path.join('./../results_TB100/OPE', tracker_name)):
                    os.mkdir(os.path.join('./../results_TB100/OPE', tracker_name))
                with open(os.path.join('./../results_TB100/OPE', tracker_name, dict['seqName'] + '.json'), 'w') as outfile:
                    json.dump(dict, outfile)
            elif mat['results']['res'].shape[1] == 6 and (tracker_name == 'ASLA' or tracker_name == 'IVT'):
                dict['res'] = mat['results']['res'][:, [0, 1, 2, -1]].tolist()
                if not os.path.exists(os.path.join('./../results_TB100/OPE', tracker_name)):
                    os.mkdir(os.path.join('./../results_TB100/OPE', tracker_name))
                with open(os.path.join('./../results_TB100/OPE', tracker_name, dict['seqName'] + '.json'), 'w') as outfile:
                    json.dump(dict, outfile)
            else:
                import warnings

                warnings.warn("WTF with %s"%(seq))
