"""
author: DI WU
stevenwudi@gmail.com
"""
from __future__ import print_function
from scripts import Temple_color_script, UAV_script
import os
import numpy as np
import json
import time
from scripts import butil
from scripts.model.result import Result
import matplotlib.pyplot as plt


# some global variables here
OVERWRITE_RESULT = False
SETUP_SEQ = False
SAVE_RESULT = True
SRC_DIR = '/home/stevenwudi/Documents/Python_Project/Temple-color-128/Temple-color-128/'
ANNO_DIR= '/home/stevenwudi/Documents/Python_Project/Temple-color-128/cfg_json/'
RESULT_SRC = './results_temple_color/{0}/'  # '{0} : OPE, SRE, TRE'



class Tracker:
    def __init__(self, name=''):
        self.name=name


def main():

    tracker = Tracker(name='KCFmulti_cnn_dsst_adapted_lr_best_valid_CNN')
    evalTypes = ['OPE']
    if SETUP_SEQ:
        print('Setup sequences ...')
        Temple_color_script.setup_seqs(SRC_DIR)

    print('Starting benchmark for trackers: {0}'.format(tracker.name))
    for evalType in evalTypes:
        seqNames = Temple_color_script.get_seq_names(SRC_DIR)
        seqs = Temple_color_script.load_seq_configs(seqNames, SRC_DIR)
        ######################################################################
        results = run_trackers(tracker, seqs, evalType)
        ######################################################################
        if len(results) > 0:
            ######################################################################
            evalResults, attrList = butil.calc_result(tracker, seqs, results, evalType, ANNO_DIR)
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
    for idxSeq in range(68, numSeq):
        subS = seqs[idxSeq]
        print('{0}:{1}, total frame: {2}'.format(idxSeq + 1,subS.name, subS.endFrame - subS.startFrame))
        ####################
        tracker, res = run_KCF_variant(tracker, subS, debug=True)
        ####################
        r = Result(tracker.name, subS.name, subS.startFrame, subS.endFrame,
                   res['type'], evalType, res['res'], res['fps'], None)
        r.refresh_dict()
        seqResults.append(r)
        # end for subseqs

    return seqResults


def run_KCF_variant(tracker, seq, debug=False):
    from keras.preprocessing import image
    from visualisation_utils import plot_tracking_result

    start_time = time.time()
    start_frame = 0
    tracker.res = []

    src = RESULT_SRC.format('OPE') + tracker.name
    if os.path.exists(os.path.join(src, seq.name + '.json')):
        json_file = os.path.join(src, seq.name + '.json')

    with open(json_file) as json_data:
        result = json.load(json_data)
        if type(result) == list:
            result = result[0]

    for frame in range(start_frame, seq.endFrame - seq.startFrame + 1):
        image_no = seq.startFrame + frame
        image_filename = seq.imgFormat.format(image_no)
        image_path = os.path.join(seq.path, image_filename)
        img_rgb = image.load_img(image_path)
        img_rgb = image.img_to_array(img_rgb)

        if debug and frame > start_frame:
            print("Frame ==", frame)
            print("pos", np.array(result['res'][frame - 1]).astype(int))
            print("gt", seq.gtRect[frame])
            print("\n")
            plot_tracking_result(frame + seq.startFrame, img_rgb, result,
                                 seq.gtRect, seq.name, wait_second=0.5)

    total_time = time.time() - start_time
    tracker.fps = len(tracker.res) / total_time
    print("Frames-per-second:", tracker.fps)

    return tracker, []

if __name__ == "__main__":
    main()
