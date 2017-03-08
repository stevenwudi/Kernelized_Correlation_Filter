"""
author: DI WU
stevenwudi@gmail.com
"""
from __future__ import print_function
from scripts import UAV_script
from KCFpy_debug import KCFTracker
import os
import numpy as np
import time
from scripts import butil
from scripts.model.result import Result


# some global variables here
OVERWRITE_RESULT = True
SETUP_SEQ = False
SAVE_RESULT = True
SRC_DIR = 'F:\Dataset_UAV123_10fps\UAV123_10fps'
RESULT_SRC = './results_UAV123/{0}/'  # '{0} : OPE, SRE, TRE'
EVAL_SEQ = 'UAV123_10fps'
IMG_DIR = os.path.join(SRC_DIR, 'data_seq', EVAL_SEQ)
ANNO_DIR = os.path.join(SRC_DIR, 'anno', EVAL_SEQ)


class Tracker:
    def __init__(self, name=''):
        self.name=name


def main():
    # tracker = KCFTracker(feature_type='multi_cnn', sub_feature_type='dsst',
    #                        sub_sub_feature_type='adapted_lr', load_model=True, vgglayer='',
    #                        model_path='./trained_models/CNN_Model_OBT100_multi_cnn_best_cifar_big_valid.h5',
    #                        name_suffix='_best_valid_CNN')
    tracker = KCFTracker(feature_type='raw',
                         optical_flow=True)
    #tracker = Tracker(name='KCFmulti_cnn_dsst_adapted_lr_best_valid_CNN')
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
        #seqs = seqs[:40]
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
    for idxSeq in range(62, numSeq):
        subS = seqs[idxSeq]
        print('{0}:{1}, total frame: {2}'.format(idxSeq + 1,subS.name, subS.endFrame - subS.startFrame))
        if not OVERWRITE_RESULT:
            trk_src = os.path.join(RESULT_SRC.format(evalType), tracker.name)
            result_src = os.path.join(trk_src, subS.name + '.json')
            if os.path.exists(result_src):
                r = UAV_script.load_seq_result(result_src)
                seqResults.append(r)
                continue
        ####################
        tracker, res = run_KCF_variant(tracker, subS)
        ####################
        r = Result(tracker.name, subS.name, subS.startFrame, subS.endFrame,
                   res['type'], evalType, res['res'], res['fps'], None)
        r.refresh_dict()
        seqResults.append(r)
        # end for subseqs
        if SAVE_RESULT:
            UAV_script.save_seq_result(RESULT_SRC, r)

    return seqResults

def run_KCF_variant(tracker, seq, debug=False):
    from keras.preprocessing import image
    from visualisation_utils import plot_tracking_rect, show_precision

    start_time = time.time()
    start_frame = 0
    tracker.res = []
    for frame in range(start_frame, seq.endFrame - seq.startFrame+1):
        image_no = seq.startFrame + frame
        _id = seq.imgFormat.format(image_no)
        image_path = os.path.join(IMG_DIR, seq.path.split('\\')[-2], _id)
        img_rgb = image.load_img(image_path)
        img_rgb = image.img_to_array(img_rgb)
        ##############
        if frame == start_frame:
            tracker.train(img_rgb, seq.gtRect[start_frame], seq.name)
        else:
            tracker.detect(img_rgb, frame)

        if debug and frame > start_frame:
            print("Frame ==", frame)
            print('horiz_delta: %.2f, vert_delta: %.2f' % (tracker.horiz_delta, tracker.vert_delta))
            print("pos", np.array(tracker.res[-1]).astype(int))
            print("gt", seq.gtRect[frame])
            print("\n")
            plot_tracking_rect(seq.name, frame, img_rgb, tracker, seq.gtRect)

    total_time = time.time() - start_time
    tracker.fps = len(tracker.res) / total_time
    print("Frames-per-second:", tracker.fps)

    if debug:
        tracker.precisions = show_precision(np.array(tracker.res), np.array(seq.gtRect), seq.name)

    res = {'type': tracker.type, 'res': tracker.res, 'fps': tracker.fps}
    return tracker, res

if __name__ == "__main__":
    main()
