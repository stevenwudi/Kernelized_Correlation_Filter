import os
import json
from model import Sequence


class Attribute:
    """
    In addition to tracking ground truth, each sequence in
    TColor-128 is also annotated by its challenge factors. Same as
    in [6], 11 factors are used for TColor-128, including illumi-
    nation variation (IV), scale variation (SV), occlusion (OCC),
    deformation (DEF), motion blur (MB), fast motion (FM), in-
    plane rotation (IPR), out-of-plane rotation (OPR), out-of-view
    (OV), background clutters (BC), and low resolution (LR). In
    particular, scale variation is decided when the ratio of the size
    of the bounding box in the current frame to that in the first
    frame falls out of the range [0.5,2]; fast motion is decided
    when the target motion is larger than 20 pixels; low resolution
    is decided when the number of pixels inside the groundtruth
    bounding box is fewer than 400 pixels. Fig. 3 gives the
    distribution of challenge factors in TColor-128. Although we
    try to make the dataset balanced in terms of challenging
    factors, trackers that handle OPR, SV, OCC, FM and IPR
    better may have some advantages over those who handle OV
    and LR better.
    """
    def __init__(self):
        self.att = ['IV', 'SV', 'OCC', 'DEF', 'MB', 'FM', 'IPR', 'OPR', 'OV', 'BC', 'LR']


def setup_seqs(SRC_DIR):
    seqs = make_seq_configs(SRC_DIR)
    for seq in seqs:
        save_seq_config(seq, SRC_DIR)


def make_seq_configs(SRC_DIR):
    names = get_seq_names(SRC_DIR)

    def load_start_end_frame(SRC_DIR):
        dict = {}
        for seqName in sorted(os.listdir(SRC_DIR)):
            fopen = os.path.join(SRC_DIR, seqName, seqName+'_frames.txt')
            with open(fopen) as f:
                content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            content_split = content[0].split(',')
            dict[seqName] = {}
            dict[seqName]['startFrame'] = int(content_split[0])
            dict[seqName]['endFrame'] = int(content_split[1])
            dict[seqName]['path'] = os.path.join(SRC_DIR, seqName, 'img')
        return dict

    seq_config = load_start_end_frame(SRC_DIR)
    seqList = []
    for name in names:
        print(name)
        gtFile = open(os.path.join(SRC_DIR, name, name+'_gt.txt'))
        gtLines = gtFile.readlines()
        gtRect = []
        for line in gtLines:
            if '\t' in line:
                gtRect.append(map(int, line.strip().split('\t')))
            elif ',' in line:
                if line[:3] == 'NaN':
                    # The NaN annotations mean that the object is either fully occluded or outside the frame.
                    gtRect.append([gtRect[-1][0], gtRect[-1][1], 0, 0])
                else:
                    gtRect.append([int(float(x)) for x in line.strip().split(',')])
            elif ' ' in line:
                gtRect.append(map(int, line.strip().split(' ')))

        # get attribute
        attribute_file = open(os.path.join(SRC_DIR, name, name+'_att.txt'))
        att_lines = attribute_file.readlines()
        attributes = [att.strip() for att in att_lines]

        init_rect = [0, 0, 0, 0]
        startFrame = int(seq_config[name]['startFrame'])
        endFrame = int(seq_config[name]['endFrame'])
        nz = 6
        ext = 'jpg'
        imgFormat = '{0:04d}.jpg'
        seq = Sequence(name, seq_config[name]['path'], startFrame, endFrame, attributes, nz, ext,
                       imgFormat, gtRect, init_rect)
        seqList.append(seq)
    return seqList


def save_seq_config(seq, SRC_DIR):
    string = json.dumps(seq.__dict__, indent=2)
    if not os.path.exists('/'.join(['/'.join(SRC_DIR.split('/')[:-1]), 'cfg_json'])):
        os.mkdir('/'.join(['/'.join(SRC_DIR.split('/')[:-1]), 'cfg_json']))
    src = os.path.join('/'.join(['/'.join(SRC_DIR.split('/')[:-1]), 'cfg_json']), seq.name+"_cfg.json")
    print(src)
    configFile = open(src, 'wb')
    configFile.write(string)
    configFile.close()


def get_seq_names(SRC_DIR):
    names = []
    for file in sorted(os.listdir(SRC_DIR)):
        # if file.endswith('_ce'):
        names.append(file)
    # the first annotation is attribute, we don't want it
    return names


def load_seq_configs(seqNames, SRC_DIR):
    return [load_seq_config(x, SRC_DIR) for x in seqNames]


def load_seq_config(seqName, SRC_DIR):
    src = os.path.join('/'.join(['/'.join(SRC_DIR.split('/')[:-1]), 'cfg_json']), seqName+"_cfg.json")
    configFile = open(src)
    string = configFile.read()
    j = json.loads(string)
    seq = Sequence(**j)
    seq.path = os.path.join(os.path.abspath(seq.path), '')
    return seq

