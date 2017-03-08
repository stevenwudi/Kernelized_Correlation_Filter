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

from KCF_CNN_RNN import KCFTracker


def main(argv):
    trackers = [KCFTracker(feature_type='multi_cnn', load_model=False)]
    #evalTypes = ['OPE', 'SRE', 'TRE']
    evalTypes = ['OPE']
    loadSeqs = 'TB100'
    try:
        opts, args = getopt.getopt(argv, "ht:e:s:", ["tracker=", "evaltype=", "sequence="])
    except getopt.GetoptError:
        print('usage : run_trackers.py -t <trackers> -s <sequences>-e <evaltypes>')
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

    print('Starting benchmark for {0} trackers, evalTypes : {1}'.format(len(trackers), evalTypes))
    for evalType in evalTypes:
        seqNames = butil.get_seq_names(loadSeqs)
        seqs = butil.load_seq_configs(seqNames)
        ######################################################################
        run_trackers(trackers, seqs, evalType)

    return 1


def run_trackers(trackers, seqs, evalType):
    tmpRes_path = RESULT_SRC.format('tmp/{0}/'.format(evalType))
    if not os.path.exists(tmpRes_path):
        os.makedirs(tmpRes_path)

    numSeq = len(seqs)

    trackerResults = dict((t, list()) for t in trackers)
    ##################################################
    # chose sequence to run from below
    ##################################################
    # we also collect data fro training here
    import h5py
    f = h5py.File("./data/OBT100_hdt_multi_cnn%d.hdf5", "w", driver="family", memb_size=2**32-1)
    X_train = f.create_dataset("x_train", (80000, 5, 240, 160), dtype='uint8', chunks=True)
    y_train = f.create_dataset("y_train", (80000, 4), dtype='float', chunks=True)
    count = 0
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
                print('{0}_{1}, {2}_{3}:{4}/{5} - {6}'.format(
                    idxTrk + 1, t.feature_type, idxSeq + 1, s.name, idx + 1, seqLen, evalType))
                rp = tmpRes_path + '_' + t.feature_type + '_' + str(idx + 1) + '/'
                if SAVE_IMAGE and not os.path.exists(rp):
                    os.makedirs(rp)
                subS = subSeqs[idx]
                subS.name = s.name + '_' + str(idx)

                ####################
                X_train, y_train, count = run_KCF_variant(t, subS, X_train, y_train, count)
                ####################
                print("count %d"%count)
                ####################

    X_train.resize(count - 1, axis=0)
    y_train.resize(count - 1, axis=0)
    f.close()
    print("done")

    return trackerResults


def run_KCF_variant(tracker, seq, X_train, y_train, count):
    from keras.preprocessing import image

    start_time = time.time()

    for frame in range(seq.endFrame - seq.startFrame):
        if frame > 0:
            img_rgb = img_rgb_next.copy()
        else:
            image_filename = seq.s_frames[frame]
            image_path = os.path.join(seq.path, image_filename)
            img_rgb = image.load_img(image_path)
            img_rgb = image.img_to_array(img_rgb)

        image_filename_next = seq.s_frames[frame+1]
        image_path_next = os.path.join(seq.path, image_filename_next)
        img_rgb_next = image.load_img(image_path_next)
        img_rgb_next = image.img_to_array(img_rgb_next)

        X_train, y_train, count = tracker.train_cnn(frame,
                          img_rgb,
                          seq.gtRect[frame],
                          img_rgb_next,
                          seq.gtRect[frame+1],
                          X_train, y_train, count
                          )

    total_time = time.time() - start_time
    tracker.fps = len(range(seq.endFrame - seq.startFrame)) / total_time
    print("Frames-per-second:", tracker.fps)

    return X_train, y_train, count


if __name__ == "__main__":
    main(sys.argv[1:])
