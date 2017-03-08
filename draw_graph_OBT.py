import matplotlib.pyplot as plt
import numpy as np
import math
from config import *
from scripts import *
RESULT_SRC = './results_{0}/{1}/' # '{0} : OPE, SRE, TRE'

def main():
    evalTypes = ['OPE']
    testname = 'tb100'
    for i in range(len(evalTypes)):
        evalType = evalTypes[i]
        result_src = RESULT_SRC.format(testname.upper(), evalType)
        trackers = os.listdir(result_src)
        scoreList = []
        for t in trackers:
            score = butil.load_scores(evalType, t, testname, result_src)
            scoreList.append(score)
        plot_graph_success(scoreList, i*2, evalType, testname)
        plot_graph_precision(scoreList, i*2+1, evalType, testname)
    plt.show()


def plot_graph_success(scoreList, fignum, evalType, testname):
    plt.figure(num=fignum, figsize=(9, 6), dpi=70)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
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
                ave = sum(attr.successRateList) /100. / float(len(attr.successRateList))
                if type(tracker) == dict:
                    if tracker['name'] == 'DSST':
                        plt.plot(thresholdSetOverlap, attr.successRateList,
                                 c=LINE_COLORS[i], label='{0} [{1:.3f}]'.format('DSST_BMVC_2014_tPAMI17', ave), lw=2.0, ls=ls)
                    elif tracker['name'] == 'MEEM':
                        plt.plot(thresholdSetOverlap, attr.successRateList,
                                 c=LINE_COLORS[i], label='{0} [{1:.3f}]'.format('MEEM_ECCV14', ave), lw=2.0,
                                 ls=ls)
                    elif tracker['name'] == 'MUSTer':
                        plt.plot(thresholdSetOverlap, attr.successRateList,
                                 c=LINE_COLORS[i], label='{0} [{1:.3f}]'.format('MUSTer_CVPR15', ave), lw=2.0, ls=ls)
                    elif tracker['name'][:3] =='HDT':
                        plt.plot(thresholdSetOverlap, attr.successRateList,
                            c = LINE_COLORS[i], label='{0} [{1:.3f}]'.format(tracker['name'].upper(), ave), lw=2.0, ls = ls)
                    elif tracker['name'] == 'KCFraw_colour':
                        plt.plot(thresholdSetOverlap, attr.successRateList,
                                 c=LINE_COLORS[i], label='{0} [{1:.3f}]'.format('KCF_ECCV12_tPAMI15', ave), lw=2.0,
                                 ls=ls)
                    # Wudi's modification:
                    else:
                        plt.plot(thresholdSetOverlap, attr.successRateList,
                            c = LINE_COLORS[i], label='{0} [{1:.3f}]'.format(tracker['name'], ave), lw=2.0, ls = ls)
                else:
                    plt.plot(thresholdSetOverlap, attr.successRateList,
                        c = LINE_COLORS[i], label='{0} [{1:.3f}]'.format(tracker, ave), lw=2.0, ls=ls)

            # else:
            #     plt.plot(thresholdSetOverlap, attr.successRateList,
            #         label='', alpha=0.5, c='#202020', ls='--')
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
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
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
                #ave = sum(attr.precisionRateList) / float(len(attr.precisionRateList))
                ave = attr.precisionRateList[20]
                if type(tracker) == dict:
                    # Wudi's modification:
                    if tracker['name']=='DSST':
                        plt.plot(thresholdSetError, attr.precisionRateList, c=LINE_COLORS[i],
                                 label='{0} [{1:.3f}]'.format('DSST_BMVC_2014_tPAMI17', ave), lw=2.0, ls=ls)
                    elif tracker['name'] == 'MEEM':
                        plt.plot(thresholdSetError, attr.precisionRateList, c=LINE_COLORS[i],
                                 label='{0} [{1:.3f}]'.format('MEEM_ECCV14', ave), lw=2.0, ls=ls)
                    elif tracker['name'] == 'MUSTer':
                        plt.plot(thresholdSetError, attr.precisionRateList, c=LINE_COLORS[i],
                                 label='{0} [{1:.3f}]'.format('MUSTer_CVPR15', ave), lw=2.0, ls=ls)
                    elif tracker['name'][:3] == 'HDT':
                        plt.plot(thresholdSetError, attr.precisionRateList, c=LINE_COLORS[i],
                                 label='{0} [{1:.3f}]'.format(tracker['name'].upper(), ave), lw=2.0, ls=ls)
                    elif tracker['name'] == 'KCFraw_colour':
                        plt.plot(thresholdSetError, attr.precisionRateList, c=LINE_COLORS[i],
                             label='{0} [{1:.3f}]'.format('KCF_ECCV12_tPAMI15', ave), lw=2.0, ls=ls)
                    else:
                        plt.plot(thresholdSetError, attr.precisionRateList,c = LINE_COLORS[i],
                                 label='{0} [{1:.3f}]'.format(tracker['name'], ave), lw=2.0, ls = ls)
                elif tracker == "HDT_cvpr2016" or tracker =='KCFvgg_rnn' or tracker[:3] =='KCF':
                    plt.plot(thresholdSetError, attr.precisionRateList,
                        c = LINE_COLORS[i], label='{0} [{1:.3f}]'.format(tracker, ave), lw=2.0, ls=ls)
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