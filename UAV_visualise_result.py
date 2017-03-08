from __future__ import print_function

"""
author: DI WU
stevenwudi@gmail.com
"""
import getopt
import sys

# some configurations files for OBT experiments, originally, I would never do that this way of importing,
# it's simple way too ugly
from scripts import UAV_script
import os
import numpy as np
import time
from scripts import butil
from scripts.model.result import Result

# some global variables here
SETUP_SEQ = True
SAVE_RESULT = True


if True:
    SRC_DIR = '/home/stevenwudi/Documents/Python_Project/UAV/UAV123/'
    RESULT_SRC = './results_UAV123/{0}/'  # '{0} : OPE, SRE, TRE'
    EVAL_SEQ = 'UAV123'
else:
    SRC_DIR = '/home/stevenwudi/Documents/Python_Project/UAV/UAV123_10fps/'
    RESULT_SRC = './results_UAV_10fps/{0}/'  # '{0} : OPE, SRE, TRE'
    EVAL_SEQ = 'UAV123_10fps'

IMG_DIR = os.path.join(SRC_DIR, 'data_seq', EVAL_SEQ)
ANNO_DIR = os.path.join(SRC_DIR, 'anno', EVAL_SEQ)


class Tracker:
    def __init__(self, name=''):
        self.name=name


def main(argv):
    #trackers = [Tracker(name='DSST')]
    trackers = [Tracker(name='KCFmulti_cnn_dsst_adapted_lr_best_valid_CNN')]

    evalTypes = ['OPE']
    loadSeqs = 'UAV123'
    try:
        opts, args = getopt.getopt(argv, "ht:e:s:", ["tracker=", "evaltype=", "sequence="])
    except getopt.GetoptError:
        print('usage : run_trackers.py -t <trackers> -s <sequences>' + '-e <evaltypes>')
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print('usage : run_trackers.py -t <trackers> -s <sequences>' + '-e <evaltypes>')
            sys.exit(0)
        elif opt in ("-t", "--tracker"):
            trackers = [x.strip() for x in arg.split(',')]
            # trackers = [arg]
        elif opt in ("-s", "--sequence"):
            loadSeqs = arg
            if loadSeqs != 'All' and loadSeqs != 'all' and \
                            loadSeqs != 'tb50' and loadSeqs != 'tb100' and \
                            loadSeqs != 'cvpr13':
                loadSeqs = [x.strip() for x in arg.split(',')]
        elif opt in ("-e", "--evaltype"):
            evalTypes = [x.strip() for x in arg.split(',')]


    print('Starting benchmark for {0} trackers, evalTypes : {1}'.format(
        len(trackers), evalTypes))
    for evalType in evalTypes:
        seqNames = UAV_script.get_seq_names(loadSeqs, ANNO_DIR)
        seqs = UAV_script.load_seq_configs(seqNames, ANNO_DIR)
        ######################################################################
        trackerResults = run_trackers(trackers, seqs, evalType)
        ######################################################################
        for tracker in trackers:
            results = trackerResults[tracker]
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
                    butil.save_scores(attrList)


def run_trackers(trackers, seqs, evalType):
    tmpRes_path = RESULT_SRC.format('tmp/{0}/'.format(evalType))
    if not os.path.exists(tmpRes_path):
        os.makedirs(tmpRes_path)

    numSeq = len(seqs)
    trackerResults = dict((t, list()) for t in trackers)
    ##################################################
    # chose sequence to run from below
    ##################################################
    for idxSeq in range(82, numSeq):
        s = seqs[idxSeq]
        subSeqs, subAnno = butil.get_sub_seqs(s, 20.0, evalType)

        for idxTrk in range(len(trackers)):
            t = trackers[idxTrk]
            seqLen = len(subSeqs)
            for idx in range(seqLen):
                subS = subSeqs[idx]
                subS.name = s.name + '_' + str(idx)

                ####################
                r_temp = Result(t.name, s.name, subS.startFrame, subS.endFrame, [], evalType, [], [], None)
                # if np.min(s.gtRect[0][2:])<12:
                #     print(s.name)
                t, res = run_KCF_variant(t, subS, r_temp, debug=True)
                ####################
            # end for tracker
    # end for allseqs
    return trackerResults


def run_KCF_variant(tracker, seq, r_temp, debug=False):
    from keras.preprocessing import image
    from visualisation_utils import plot_tracking_result
    import json
    start_time = time.time()
    start_frame = 0
    tracker.res = []

    src = RESULT_SRC.format('OPE') + tracker.name
    if os.path.exists(os.path.join(src, r_temp.seqName+'.json')):
        json_file = os.path.join(src, r_temp.seqName+'.json')
    else:
        json_file = os.path.join(src, (r_temp.seqName+'.json').lower())

    with open(json_file) as json_data:
        result = json.load(json_data)
        if type(result) == list:
            result = result[0]

    for frame in range(start_frame, seq.endFrame - seq.startFrame+1):
        image_filename = seq.s_frames[frame]
        image_path = os.path.join(seq.path, image_filename)
        img_rgb = image.load_img(image_path)
        img_rgb = image.img_to_array(img_rgb)

        if debug and frame > start_frame:
            print("Frame ==", frame)
            print("pos", np.array(result['res'][frame-1]).astype(int))
            print("gt", seq.gtRect[frame])
            print("\n")
            plot_tracking_result(frame, img_rgb, result, seq.gtRect, r_temp.seqName, wait_second=0.5)

    total_time = time.time() - start_time
    tracker.fps = len(tracker.res) / total_time
    print("Frames-per-second:", tracker.fps)

    return tracker, []

if __name__ == "__main__":
    main(sys.argv[1:])
