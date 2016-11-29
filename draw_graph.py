import matplotlib.pyplot as plt
import numpy as np
import math
from config import *
from scripts import *


def main():
    evalTypes = ['OPE']
    testname = 'tb50'
    for i in range(len(evalTypes)):
        evalType = evalTypes[i]
        result_src = RESULT_SRC.format(evalType)
        trackers = os.listdir(result_src)
        scoreList = []
        for t in trackers:
            score = butil.load_scores(evalType, t, testname)
            scoreList.append(score)
        plot_graph_success(scoreList, i*2, evalType, testname)
        plot_graph_precision(scoreList, i*2+1, evalType, testname)
    plt.show()


def plot_graph_success(scoreList, fignum, evalType, testname):
    plt.figure(num=fignum, figsize=(9, 6), dpi=70)
    rankList = sorted(scoreList,  key=lambda o: sum(o[0].successRateList), reverse=True)
    for i in range(len(rankList)):
        result = rankList[i]
        tracker = result[0].tracker
        attr = result[0]
        if len(attr.successRateList) == len(thresholdSetOverlap):
            if i < MAXIMUM_LINES:
                ls = '-'
                if i % 2 == 1:
                    ls = '--'
                ave = sum(attr.successRateList) / float(len(attr.successRateList))
                if type(tracker) == dict:
                    # Wudi's modification:
                    plt.plot(thresholdSetOverlap, attr.successRateList,
                        c = LINE_COLORS[i], label='{0} [{1:.2f}]'.format(tracker['name'], ave), lw=2.0, ls = ls)
                else:
                    plt.plot(thresholdSetOverlap, attr.successRateList,
                        c = LINE_COLORS[i], label='{0} [{1:.2f}]'.format(tracker, ave), lw=2.0, ls=ls)
            else:
                plt.plot(thresholdSetOverlap, attr.successRateList, 
                    label='', alpha=0.5, c='#202020', ls='--')
        else:
            print('err')
    plt.title('Success plots of {0}_{1} (sequence average)'.format(evalType, testname.upper()))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('thresholds')
    plt.xticks(np.arange(thresholdSetOverlap[0], thresholdSetOverlap[len(thresholdSetOverlap)-1]+0.1, 0.1))
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    # plt.savefig(BENCHMARK_SRC + 'graph/{0}_sq.png'.format(evalType), dpi=74, bbox_inches='tight')
    return plt


def plot_graph_precision(scoreList, fignum, evalType, testname):

    plt.figure(num=fignum, figsize=(9, 6), dpi=70)
    # some don't have precison list--> we will delete them?
    for t in scoreList:
        if len(t[0].precisionRateList)<20:
            print(t[0].tracker)
            t[0].precisionRateList= np.zeros(51)
            t[0].precisionRateList[20] = 0

    rankList = sorted(scoreList,  key=lambda o: o[0].precisionRateList[20], reverse=True)
    for i in range(len(rankList)):
        result = rankList[i]
        tracker = result[0].tracker
        attr = result[0]
        if len(attr.precisionRateList) == len(thresholdSetError):
            if i < MAXIMUM_LINES:
                ls = '-'
                if i % 2 == 1:
                    ls = '--'
                ave = sum(attr.precisionRateList) / float(len(attr.precisionRateList))
                if type(tracker) == dict:
                    # Wudi's modification:
                    plt.plot(thresholdSetError, attr.precisionRateList,
                        c = LINE_COLORS[i], label='{0} [{1:.2f}]'.format(tracker['name'], ave), lw=2.0, ls = ls)
                elif tracker == "HDT_cvpr2016" or tracker =='KCFvgg_rnn' or tracker =='KCFmulti_cnn':
                    plt.plot(thresholdSetError, attr.precisionRateList,
                        c = LINE_COLORS[i], label='{0} [{1:.2f}]'.format(tracker, ave), lw=2.0, ls=ls)
            # else:
            #     plt.plot(thresholdSetOverlap, attr.precisionRateList,
            #         label='', alpha=0.5, c='#202020', ls='--')
        else:
            print 'err'
    plt.title('Precision plots of {0}_{1} (sequence average)'.format(evalType, testname.upper()))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('thresholds')
    plt.xticks(np.arange(thresholdSetError[0], thresholdSetError[len(thresholdSetError)-1], 10))
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    # plt.savefig(BENCHMARK_SRC + 'graph/{0}_sq.png'.format(evalType), dpi=74, bbox_inches='tight')
    plt.show()
    return plt


if __name__ == '__main__':
    main()