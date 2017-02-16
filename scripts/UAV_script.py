import os
import sys
import json
from model import Sequence
from scripts.model.result import Result


class Attribute:
    """
    IV Illumination Variation: the illumination of the target changes significantly.
    SV Scale Variation: the ratio of initial and at least one subsequent bounding box is outside the range [0.5, 2].
    POC Partial Occlusion: the target is partially occluded.
    FOC Full Occlusion: the target is fully occluded.
    OV Out-of-View: some portion of the target leaves the view.
    FM Fast Motion: motion of the ground-truth bounding box is larger than 20 pixels between two consecutive frames.
    CM Camera Motion: abrupt motion of the camera.
    BC Background Clutter: the background near the target has similar appearance as the target.
    SOB Similar Object: there are objects of similar shape or same type near the target.
    ARC
    Aspect Ratio Change: the fraction of ground-truth aspect ratio in the first frame and at least one subsequent frame is outside the range [0.5,
    2].
    VC Viewpoint Change: viewpoint affects target appearance significantly.
    LR Low Resolution: at least one ground-truth bounding box has less than 400 pixels.
    """
    def __init__(self):
        self.att = ['SV', 'ARC', 'LR', 'FM', 'FOC', 'POC', 'OV', 'BC', 'IV', 'VC', 'CM', 'SOB']


def load_start_end_frame(SRC_DIR):
    import scipy.io as sio
    # we load the .mat seq config file
    f = sio.loadmat(os.path.join(SRC_DIR, 'configSeqs.mat'))

    dict ={}
    for i in range(len(f['seqs'][0])):
        seqName = f['seqs'][0][i][0][0][0][0]
        dict[seqName] = {}
        dict[seqName]['startFrame'] = f['seqs'][0][i][0][0][2][0][0]
        dict[seqName]['endFrame'] = f['seqs'][0][i][0][0][3][0][0]
        path = f['seqs'][0][i][0][0][1][0].split('\\')
        dict[seqName]['path'] = path[-2]
    return dict


def setup_seqs(loadSeqs, SRC_DIR, ANNO_DIR, IMG_DIR):

    seqs = make_seq_configs(loadSeqs, SRC_DIR, ANNO_DIR, IMG_DIR)
    for seq in seqs:
        save_seq_config(seq, ANNO_DIR)


def save_seq_config(seq, ANNO_DIR):
    string = json.dumps(seq.__dict__, indent=2)
    if not os.path.exists(os.path.join(ANNO_DIR, 'json')):
        os.mkdir(os.path.join(ANNO_DIR, 'json'))
    src = os.path.join(ANNO_DIR, 'json', seq.name+"_cfg.json")
    print(src)
    configFile = open(src, 'wb')
    configFile.write(string)
    configFile.close()


def make_seq_configs(loadSeqs, SRC_DIR, ANNO_DIR, IMG_DIR):
    names = get_seq_names(loadSeqs, ANNO_DIR)
    seq_config = load_start_end_frame(SRC_DIR)
    seqList = []
    attribute_all = Attribute()
    for name in names:
        gtFile = open(os.path.join(ANNO_DIR, name+'.txt'))
        gtLines = gtFile.readlines()
        gtRect = []
        for line in gtLines:
            if '\t' in line:
                gtRect.append(map(int, line.strip().split('\t')))
            elif ',' in line:
                if line[:3] == 'NaN':
                    # The NaN annotations mean that the object is either fully occluded or outside the frame.
                    gtRect.append([gtRect[-1][0], gtRect[-1][1], 0,0])
                else:
                    gtRect.append(map(int, line.strip().split(',')))
            elif ' ' in line:
                gtRect.append(map(int, line.strip().split(' ')))

        # get attribute
        attribute_file = open(os.path.join(ANNO_DIR, 'att', name+'.txt'))
        att_lines = attribute_file.readlines()
        attributes = []
        for i, f in enumerate(att_lines[0].split(',')):
            if f[0] == '1':
                attributes.append(attribute_all.att[i])

        init_rect = [0, 0, 0, 0]
        path = os.path.join(IMG_DIR, seq_config[name]['path'])
        startFrame = int(seq_config[name]['startFrame'])
        endFrame = int(seq_config[name]['endFrame'])
        nz = 6
        ext = 'jpg'
        imgFormat= '{0:06d}.jpg'
        seq = Sequence(name, path, startFrame, endFrame, attributes, nz, ext,
                       imgFormat, gtRect, init_rect)
        seqList.append(seq)
    return seqList


def get_seq_names(loadSeqs, ANNO_DIR):
    if loadSeqs == 'UAV123':
        names = []
        for file in os.listdir(ANNO_DIR):
            if file.endswith('.txt'):
                names.append(file)
        names = sorted([x[:-4] for x in names])
        # the first annotation is attribute, we don't want it
        return names


def load_seq_configs(seqNames, ANNO_DIR):
    return [load_seq_config(x, ANNO_DIR) for x in seqNames]


def load_seq_config(seqName, ANNO_DIR):
    src = os.path.join(ANNO_DIR, 'json', seqName + "_cfg.json")
    configFile = open(src)
    string = configFile.read()
    j = json.loads(string)
    seq = Sequence(**j)
    seq.path = os.path.join(os.path.abspath(seq.path), '')
    return seq


def save_seq_result(RESULT_SRC, result):
    tracker = result.tracker
    seqName = result.seqName
    evalType = result.evalType
    src = RESULT_SRC.format(evalType) + tracker
    if not os.path.exists(src):
        os.makedirs(src)
    try:
        string = json.dumps(result, default=lambda o : o.__dict__)
    except:
        print map(type, result[0].__dict__.values())
        sys.exit()
    fileName = src + '/{0}.json'.format(seqName)
    resultFile = open(fileName, 'wb')
    resultFile.write(string)
    resultFile.close()


def load_seq_result(result_src):
    resultFile = open(result_src)
    string = resultFile.read()
    jsonList = json.loads(string)
    if type(jsonList) is list:
        return [Result(**j) for j in jsonList]
    elif type(jsonList) is dict:
        return [Result(**jsonList)]
    return None


def save_scores(scoreList, RESULT_SRC, testname=None):
    tracker = scoreList[0].tracker
    evalType = scoreList[0].evalType
    trkSrc = RESULT_SRC.format(evalType) + tracker
    if testname == None:
        scoreSrc = trkSrc + '/scores'
    else:
        scoreSrc = trkSrc + '/scores_{0}'.format(testname)
    if not os.path.exists(scoreSrc):
        os.makedirs(scoreSrc)
    for score in scoreList:
        if score.tracker == 'KCF':
            score.tracker = score.tracker
            # the following two attributes can not be copied, we need to delete them
            # if hasattr(score.tracker, 'extract_model'):
            #     del score.tracker.extract_model
            # if hasattr(score.tracker, 'base_model'):
            #     del score.tracker.base_model
        string = json.dumps(score, default=lambda o : o.__dict__)
        fileName = scoreSrc + '/{0}.json'.format(score.name)
        scoreFile = open(fileName, 'wb')
        scoreFile.write(string)
    scoreFile.close()

