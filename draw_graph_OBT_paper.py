import matplotlib.pyplot as plt
from scripts import *
# from matplotlib import rc
# # activate latex text rendering
# rc('text', usetex=True)


def main():
    evalTypes = ['OPE']
    testname = 'OBT100'
    for i in range(len(evalTypes)):
        evalType = evalTypes[i]
        result_src = './results_OBT_100_paper/'
        trackers = os.listdir(result_src)
        scoreList = []
        for t in trackers:
            score = butil.load_scores(evalType, t, testname, result_src)
            score.append(t)
            if score:
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
                if result[-1] == 'KMC':
                    # label_str = '{0} [{1:.3f}]'.format(result[-1], ave)
                    # plt.plot(thresholdSetOverlap, attr.successRateList,
                    #      c=LINE_COLORS[i], label="$\it{ture}$", lw=2.0, ls=ls)
                    plt.plot(thresholdSetOverlap, attr.successRateList,
                         c='r', label='{0} [{1:.3f}]'.format(result[-1], ave), lw=8.0, ls='-')
                else:
                    plt.plot(thresholdSetOverlap, attr.successRateList,
                         c=LINE_COLORS[i], label='{0} [{1:.3f}]'.format(result[-1], ave), lw=2.0, ls=ls)

    #plt.title('Success plots of {0}_{1} (sequence average)'.format(evalType, testname.upper()))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('thresholds')
    plt.xticks(np.arange(thresholdSetOverlap[0], thresholdSetOverlap[len(thresholdSetOverlap)-1]+0.1, 0.1))
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    plt.savefig('cvprw_paper_figure/OBT_100_success.png', dpi=140, bbox_inches='tight')
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
                if result[-1] == 'KMC':
                    # label_str = '{0} [{1:.3f}]'.format(result[-1], ave)
                    # plt.plot(thresholdSetOverlap, attr.successRateList,
                    #      c=LINE_COLORS[i], label="$\it{ture}$", lw=2.0, ls=ls)
                    plt.plot(thresholdSetError, attr.precisionRateList,
                         c='r', label='{0} [{1:.3f}]'.format(result[-1], ave), lw=8.0, ls='-')
                else:
                    plt.plot(thresholdSetError, attr.precisionRateList, c=LINE_COLORS[i],
                         label='{0} [{1:.3f}]'.format(result[-1], ave), lw=2.0, ls=ls)

    #plt.title('Precision plots of {0}_{1} (sequence average)'.format(evalType, testname.upper()))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('thresholds')
    plt.xticks(np.arange(thresholdSetError[0], thresholdSetError[len(thresholdSetError)-1], 10))
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    plt.savefig('cvprw_paper_figure/OBT_100_precision.png', dpi=74, bbox_inches='tight')
    plt.show()
    return plt


if __name__ == '__main__':
    main()