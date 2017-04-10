"""
author: DI WU
stevenwudi@gmail.com
"""
from __future__ import print_function
from scripts import UAV_script

import os
import numpy as np
import time
from scripts import butil
from scripts.model.result import Result


# some global variables here
SAVE_RESULT = True
SRC_DIR = '/home/stevenwudi/Documents/Python_Project/UAV/UAV123/'
RESULT_SRC = './results_UAV123/{0}/'  # '{0} : OPE, SRE, TRE'
EVAL_SEQ = 'UAV123'
IMG_DIR = os.path.join(SRC_DIR, 'data_seq', EVAL_SEQ)
ANNO_DIR = os.path.join(SRC_DIR, 'anno', EVAL_SEQ)
SETUP_SEQ = False
OVERWRITE_RESULT = True

Tracker_names = ['ASLA', 'CSK', 'DSST', 'IVT', 'KCF_GaussHog', 'KCF_LinearGray', 'KCF_LinearHog', 'MEEM', 'MUSTER',
                   'OAB', 'SAMF', 'SRDCF', 'Struck', 'TLD']

class Tracker:
    def __init__(self, name=''):
        self.name=name


def main():

    for name in Tracker_names:
        tracker = Tracker(name=name)
        # evalTypes = ['OPE', 'SRE', 'TRE']
        evalTypes = ['OPE']
        loadSeqs = 'UAV123'

        if SETUP_SEQ:
            print('Setup sequences ...')
            UAV_script.setup_seqs(loadSeqs, SRC_DIR,  ANNO_DIR, IMG_DIR)

        print('Starting benchmark for trackers: {0}, evalTypes : {1}'.format(tracker.name, evalTypes))
        for evalType in evalTypes:
            seqNames = UAV_script.get_seq_names(loadSeqs, ANNO_DIR)
            seqs = UAV_script.load_seq_configs(seqNames, ANNO_DIR)
            ######################################################################
            results = run_trackers(tracker, seqs, evalType)
            ######################################################################
            if len(results) > 0:
                ######################################################################
                evalResults, attrList = butil.calc_result(tracker, seqs, results, evalType, SRC_DIR)
                ######################################################################
                print ("Result of Sequences\t -- '{0}'".format(tracker.name))
                for i, seq in enumerate(seqs):
                    try:
                        print('\t{0}:\'{1}\'{2}\taveCoverage : {3:.3f}%\taveErrCenter : {4:.3f}'.format(
                            i,
                            seq.name,
                            " " * (12 - len(seq.name)),
                            sum(seq.aveCoverage) / len(seq.aveCoverage) * 100,
                            sum(seq.aveErrCenter) / len(seq.aveErrCenter)))
                    except:
                        print('\t\'{0}\'  ERROR!!'.format(seq.name))

                print("Result of attributes\t -- '{0}'".format(tracker.name))
                for attr in attrList:
                    print("\t\'{}\'\t overlap : {:04.2f}% \t\t failures : {:04.2f}".format(attr.name, attr.overlap, attr.error))

                if SAVE_RESULT:
                    UAV_script.save_scores(attrList, RESULT_SRC)


def run_trackers(tracker, seqs, evalType):
    numSeq = len(seqs)
    seqResults = []
    ##################################################
    # chose sequence to run from below
    ##################################################
    for idxSeq in range(0, numSeq):
        subS = seqs[idxSeq]
        print('{0}:{1}, total frame: {2}'.format(idxSeq + 1, subS.name, subS.endFrame - subS.startFrame))
        trk_src = os.path.join(RESULT_SRC.format(evalType), tracker.name)
        result_src = os.path.join(trk_src, subS.name + '.json')
        if os.path.exists(result_src):
            r = UAV_script.load_seq_result(result_src)
            seqResults.append(r)
            continue

    return seqResults


if __name__ == "__main__":
    main()
