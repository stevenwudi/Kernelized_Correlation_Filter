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

from KCFpy_debug import KCFTracker


def main(argv):
    trackers = [KCFTracker(feature_type='vgg_rnn')]
    #evalTypes = ['OPE', 'SRE', 'TRE']
    evalTypes = ['OPE']
    loadSeqs = 'TB50'
    try:
        opts, args = getopt.getopt(argv, "ht:e:s:", ["tracker=", "evaltype=", "sequence="])
    except getopt.GetoptError:
        print 'usage : run_trackers.py -t <trackers> -s <sequences>' + '-e <evaltypes>'
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print 'usage : run_trackers.py -t <trackers> -s <sequences>' + '-e <evaltypes>'
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
        print 'Setup sequences ...'
        butil.setup_seqs(loadSeqs)

    print 'Starting benchmark for {0} trackers, evalTypes : {1}'.format(
        len(trackers), evalTypes)
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
                evalResults, attrList = butil.calc_result(tracker, seqs, results, evalType)
                ######################################################################
                print "Result of Sequences\t -- '{0}'".format(tracker.name)
                for seq in seqs:
                    try:
                        print '\t\'{0}\'{1}'.format(
                            seq.name, " " * (12 - len(seq.name))),
                        print "\taveCoverage : {0:.3f}%".format(
                            sum(seq.aveCoverage) / len(seq.aveCoverage) * 100),
                        print "\taveErrCenter : {0:.3f}".format(
                            sum(seq.aveErrCenter) / len(seq.aveErrCenter))
                    except:
                        print '\t\'{0}\'  ERROR!!'.format(seq.name)

                print "Result of attributes\t -- '{0}'".format(tracker.name)
                for attr in attrList:
                    print "\t\'{0}\'".format(attr.name),
                    print "\toverlap : {0:02.1f}%".format(attr.overlap),
                    print "\tfailures : {0:.1f}".format(attr.error)

                if SAVE_RESULT:
                    butil.save_scores(attrList)


def run_trackers(trackers, seqs, evalType, shiftTypeSet):
    tmpRes_path = RESULT_SRC.format('tmp/{0}/'.format(evalType))
    if not os.path.exists(tmpRes_path):
        os.makedirs(tmpRes_path)

    numSeq = len(seqs)

    trackerResults = dict((t, list()) for t in trackers)
    frame_list = []
    w_list = []
    h_list = []
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
            seqLen = len(subSeqs)
            for idx in range(seqLen):
                print '{0}_{1}, {2}_{3}:{4}/{5} - {6}'.format(
                    idxTrk + 1, t.feature_type, idxSeq + 1, s.name, idx + 1, seqLen, evalType)
                rp = tmpRes_path + '_' + t.feature_type + '_' + str(idx + 1) + '/'
                if SAVE_IMAGE and not os.path.exists(rp):
                    os.makedirs(rp)
                subS = subSeqs[idx]
                subS.name = s.name + '_' + str(idx)

                ####################
                frame_num, w, h = run_KCF_variant(t, subS, debug=False)
                ####################
                frame_list.append(frame_num)
                w_list.append(w)
                h_list.append(h)
            # end for tracker
    # end for allseqs
    frame_list = np.asarray(frame_list)
    w_list = np.asarray(w_list)
    h_list = np.asarray(h_list)
    print("max: %d, min: %d, mean: %d"%(frame_list.max(), frame_list.min(), frame_list.mean()))
    print("max: %d, min: %d, mean: %d" % (w_list.max(), w_list.min(), w_list.mean()))
    print("max: %d, min: %d, mean: %d" % (h_list.max(), h_list.min(), h_list.mean()))

    return trackerResults


def run_KCF_variant(tracker, seq, debug=False):

    tracker.res = []
    frame_num = len(range(seq.endFrame - seq.startFrame+1))
    w = 0
    h = 0
    for gt in seq.gtRect:
        w += gt[2]
        h += gt[3]

    w = w * 1.0 / frame_num
    h = h * 1.0 / frame_num

    return frame_num, w, h

if __name__ == "__main__":
    main(sys.argv[1:])
