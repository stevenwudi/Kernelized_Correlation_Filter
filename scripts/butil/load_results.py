from config import *
from scripts import *
import json, sys

def save_seq_result(result):
    tracker = result[0].tracker
    seqName = result[0].seqName
    evalType = result[0].evalType
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

def save_scores(scoreList, testname=None):
    tracker = scoreList[0].tracker
    evalType = scoreList[0].evalType
    trkSrc = RESULT_SRC.format(evalType) + tracker.name
    if testname == None:
        scoreSrc = trkSrc + '/scores'
    else:
        scoreSrc = trkSrc + '/scores_{0}'.format(testname)
    if not os.path.exists(scoreSrc):
        os.makedirs(scoreSrc)
    for score in scoreList:
        if score.tracker.name[:3] == 'KCF':
            score.tracker = score.tracker.name
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


def save_scores_new_tracker(scoreList, tracker_name='HDT_cvpr2016'):

    evalType = scoreList[0].evalType
    trkSrc = RESULT_SRC.format(evalType) + tracker_name
    scoreSrc = trkSrc + '/scores'
    if not os.path.exists(scoreSrc):
        os.makedirs(scoreSrc)
    for score in scoreList:
        string = json.dumps(score, default=lambda o : o.__dict__)
        fileName = scoreSrc + '/{0}.json'.format(score.name)
        scoreFile = open(fileName, 'wb')
        scoreFile.write(string)
    scoreFile.close()


def load_all_results(evalType):
    resultSRC = RESULT_SRC.format(evalType)
    trackers = os.listdir(resultSRC)
    resultList = dict()
    for tracker in trackers:
        results, attrs = load_result(evalType, tracker)
        resultList[tracker] = (results, attrs)

    return resultList

def load_result(evalType, tracker):
    resultSRC = RESULT_SRC.format(evalType)
    print 'Loading \'{0}\'...'.format(tracker),
    src = os.path.join(resultSRC, tracker)
    resultNames = os.listdir(src)
    attrs = []
    results = []
    for name in resultNames:
        if name == 'attributes':
            attrSrc = os.path.join(src, name)
            attrNames = os.listdir(attrSrc)
            for attrName in attrNames:
                attrFile = open(os.path.join(attrSrc, attrName))
                string = attrFile.read()
                j = json.loads(string)
                attr = Attribute(**j)
                attr.successRateList = map(lambda o:o*100, attr.successRateList)
                attrs.append(attr)
                attrs.sort()
        elif name.endswith('.json'):
            resultFile = open(os.path.join(src, name))
            string = resultFile.read()
            jsonList = json.loads(string)
            if type(jsonList) is list:
                results.append([Result(**j) for j in jsonList])
            elif type(jsonList) is dict:
                results.append([Result(**jsonList)])
    print '({0} seqs)'.format(len(resultNames) - 1)
    return results, attrs

def load_seq_result(evalType, tracker, sequence):
    resultSRC = RESULT_SRC.format(evalType)
    print 'Loading {0}/{1}...'.format(tracker.name, sequence)
    src = os.path.join(resultSRC, tracker.name)
    result_src = os.path.join(src, sequence+'.json')
    resultFile = open(result_src)
    string = resultFile.read()
    jsonList = json.loads(string)
    if type(jsonList) is list:
        return [Result(**j) for j in jsonList]
    elif type(jsonList) is dict:
        return [Result(**jsonList)]
    return None

def load_all_scores(evalType, testname):
    resultSRC = RESULT_SRC.format(evalType)
    trackers = os.listdir(resultSRC)
    attrList = [(t, load_scores(evalType, t, testname)) for t in trackers]
    return attrList

def load_scores(evalType, tracker, testname):
    resultSRC = RESULT_SRC.format(evalType)
    print 'Loading \'{0}\'...'.format(tracker)
    if tracker[:3] == 'KCF' or tracker[:4] == 'DSST' or tracker[:4]=='MEEM'\
            or tracker[:4]=='MUST' or tracker[:3]=='HDT':
        src = os.path.join(resultSRC, tracker + '/scores')
    else:
        src = os.path.join(resultSRC, tracker + '/scores_{0}'.format(testname))
    attrNames = os.listdir(src)
    attrs = []
    for attrName in attrNames:
        attrFile = open(os.path.join(src, attrName))
        string = attrFile.read()
        j = json.loads(string)
        attr = Score(**j)
        attr.successRateList = map(lambda o:o*100, attr.successRateList)
        attrs.append(attr)
        attrs.sort()
    return attrs