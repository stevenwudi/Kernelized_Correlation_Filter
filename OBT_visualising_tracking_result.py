from __future__ import print_function

"""
author: DI WU
stevenwudi@gmail.com
"""
import getopt
import sys

# some configurations files for OBT experiments, originally, I would never do that this way of importing,
# it's simple way too ugly
from config import *
from scripts import *


OVERWRITE_RESULT = True


class Tracker:
    def __init__(self, name=''):
        self.name=name


def main(argv):
    #trackers = [Tracker(name='DSST')]
    trackers = [Tracker(name='KCFmulti_cnn_dsst_adapted_lr_best_valid_CNN')]
    #trackers = [Tracker(name='KCFmulti_cnn_dsst_adapted_lr_hdtadl_0.002')]
    #trackers = [Tracker(name='KCFmulti_cnn_dsst_adapted_lr_grabcut_xcorr0.5')]

    evalTypes = ['OPE']
    loadSeqs = 'TB100'
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

    if SETUP_SEQ:
        print('Setup sequences ...')
        butil.setup_seqs(loadSeqs)

    print('Starting benchmark for {0} trackers, evalTypes : {1}'.format(
        len(trackers), evalTypes))
    for evalType in evalTypes:
        seqNames = butil.get_seq_names(loadSeqs)
        seqs = butil.load_seq_configs(seqNames)
        ######################################################################
        trackerResults = run_trackers(trackers, seqs, evalType, shiftTypeSet)
        ######################################################################
        for tracker in trackers:
            results = trackerResults[tracker]
            if len(results) > 0:
                ######################################################################
                evalResults, attrList = butil.calc_result(tracker, seqs, results, evalType, SEQ_SRC)
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


def run_trackers(trackers, seqs, evalType, shiftTypeSet):
    tmpRes_path = RESULT_SRC.format('tmp/{0}/'.format(evalType))
    if not os.path.exists(tmpRes_path):
        os.makedirs(tmpRes_path)

    numSeq = len(seqs)
    trackerResults = dict((t, list()) for t in trackers)
    ##################################################
    # chose sequence to run from below
    ##################################################
    for idxSeq in range(53, numSeq):
        s = seqs[idxSeq]
        subSeqs, subAnno = butil.get_sub_seqs(s, 20.0, evalType)

        for idxTrk in range(len(trackers)):
            t = trackers[idxTrk]

            if not OVERWRITE_RESULT:

                trk_src = os.path.join(RESULT_SRC.format(evalType), t.name)
                result_src = os.path.join(trk_src, s.name + '.json')
                if os.path.exists(result_src):
                    seqResults = butil.load_seq_result(evalType, t, s.name)
                    trackerResults[t].append(seqResults)
                    continue
            seqLen = len(subSeqs)
            for idx in range(seqLen):
                subS = subSeqs[idx]
                subS.name = s.name + '_' + str(idx)

                ####################
                r_temp = Result(t.name, s.name, subS.startFrame, subS.endFrame, [], evalType, [], [], None)
                t, res = run_KCF_variant(t, subS, r_temp, debug=True)
                return 0
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
            plot_tracking_result(frame + seq.startFrame, img_rgb, result, seq.gtRect, r_temp.seqName, wait_second=0.1)

    total_time = time.time() - start_time
    tracker.fps = len(tracker.res) / total_time
    print("Frames-per-second:", tracker.fps)

    return tracker, []

if __name__ == "__main__":
    main(sys.argv[1:])
