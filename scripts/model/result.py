from collections import OrderedDict
from config import *

class Result:
    # tracker : trakcer name
    # seqName : Sequence name
    # startFrame : start frame number
    # endFrame : end frame number
    # res : results 
    # resType : result type
    # fps

    def __init__(self, tracker, seqName, startFrame=0, endFrame=0,
        resType='rect', evalType='OPE', res=[], fps=0, shiftType=None, tmplsize=None):
        self.tracker = tracker
        self.seqName = seqName
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.resType = resType
        self.evalType = evalType
        self.res = res
        self.fps = fps
        self.shiftType = shiftType
        self.tmplsize = tmplsize
        
        self.__dict__ = OrderedDict([
            ('tracker', self.tracker),
            ('seqName', self.seqName),
            ('startFrame', self.startFrame),
            ('endFrame', self.endFrame),
            ('evalType', self.evalType),
            ('fps', self.fps),
            ('shiftType', self.shiftType),
            ('resType', self.resType),
            ('tmplsize', self.tmplsize),
            ('res', self.res)])

    def refresh_dict(self):
        self.__dict__ = OrderedDict([
            ('tracker', self.tracker),
            ('seqName', self.seqName),
            ('startFrame', self.startFrame),
            ('endFrame', self.endFrame),
            ('evalType', self.evalType),
            ('fps', self.fps),
            ('shiftType', self.shiftType),
            ('resType', self.resType),
            ('tmplsize', self.tmplsize),
            ('res', self.res)])
        
        
