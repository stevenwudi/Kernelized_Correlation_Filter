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
OVERWRITE_RESULT = False


class Tracker:
    def __init__(self, name=''):
        self.name=name


def main(argv):
    # tracker = KCFTracker(feature_type='multi_cnn', sub_feature_type='dsst',
    #                        sub_sub_feature_type='adapted_lr', load_model=True, vgglayer='',
    #                        model_path='./trained_models/CNN_Model_OBT100_multi_cnn_best_cifar_big_valid.h5',
    #                        name_suffix='_best_valid_CNN')
    trackers = [Tracker(name='KCFmulti_cnn_dsst_adapted_lr_hdt')]
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
    for idxSeq in range(0, numSeq):
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
            seqResults = []
            seqLen = len(subSeqs)
            # end for subseqs
            if SAVE_RESULT:
                butil.save_seq_result(seqResults)

            trackerResults[t].append(seqResults)
            # end for tracker
    # end for allseqs
    return trackerResults


if __name__ == "__main__":
    main(sys.argv[1:])
