

import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

# utility functions
from mpl_toolkits.axes_grid1 import make_axes_locatable


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    # mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
    #                         ncols * imshape[1] + (ncols - 1) * border),
    #                        dtype=np.float32)
    if len(imgs.shape) == 4:
        mosaic = np.zeros((nrows * imshape[0] + (nrows - 1) * border,
                           ncols * imshape[1] + (ncols - 1) * border, 3),
                          dtype=np.float32)
    else:
        mosaic = np.zeros((nrows * imshape[0] + (nrows - 1) * border,
                                ncols * imshape[1] + (ncols - 1) * border),
                               dtype=np.float32)


    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in xrange(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
        col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic

    # plt.imshow(make_mosaic(np.random.random((9, 10, 10)), 3, 3, border=1))


def plot_tracking_rect(frame, img_rgb, tracker, gtRect):
    from matplotlib.patches import Rectangle
    plt.figure(1)
    plt.clf()

    # Because of PIL read image
    if img_rgb.shape[0] == 3:
        img_rgb = img_rgb.transpose(1, 2, 0)/255.

    tracking_figure_axes = plt.subplot(221)
    tracking_rect = Rectangle(
        xy=(tracker.res[-1][0], tracker.res[-1][1]),
        width=tracker.target_sz[1],
        height=tracker.target_sz[0],
        facecolor='none',
        edgecolor='r',
        )

    gt_rect = Rectangle(
        xy=(gtRect[frame - 1][0], gtRect[frame - 1][1]),
        width=gtRect[frame - 1][2],
        height=gtRect[frame - 1][3],
        facecolor='none',
        edgecolor='g',
    )

    # if tracker.feature_type == 'vgg_rnn':
    #     old_gt_rect = Rectangle(
    #         xy=(tracker.pos_old[0], tracker.pos_old[1]),
    #         width=gtRect[frame - 1][2],
    #         height=gtRect[frame - 1][3],
    #         facecolor='none',
    #         edgecolor='b',
    #     )
    #     tracking_figure_axes.add_patch(old_gt_rect)

    tracking_figure_axes.add_patch(tracking_rect)
    tracking_figure_axes.add_patch(gt_rect)

    import matplotlib.patches as mpatches

    # red_label = mpatches.Patch(color='red', label='tracking result')
    # green_label = mpatches.Patch(color='green', label='gt')
    # plt.legend(handles=[red_label, green_label])
    plt.imshow(img_rgb)
    plt.title('frame: %d' % frame)

    plt.subplot(222)
    if tracker.feature_type == 'vgg' or tracker.feature_type == 'resnet50' or tracker.feature_type == 'vgg_rnn' \
            or tracker.feature_type == 'cnn' or tracker.feature_type == 'multi_cnn':
        tracker.im_crop = tracker.im_crop.transpose(1, 2, 0) / 255.
    # if tracker.feature_type == 'cnn':
    #     im_crop = tracker.im_crop / 255.
    plt.imshow(tracker.im_crop)
    plt.title('Current scale factor %0.3f' % tracker.currentScaleFactor)

    plt.subplot(223)
    if tracker.feature_type == 'vgg' or tracker.feature_type == 'resnet50' or tracker.feature_type == 'vgg_rnn' or tracker.feature_type == 'cnn':
        features = tracker.x.transpose(2, 0, 1) / tracker.x.max()
        plt.imshow(make_mosaic(features[:9], 3, 3, border=1))
    elif tracker.feature_type == 'multi_cnn':
        features = tracker.x[0].transpose(2, 0, 1) / tracker.x[0].max()
        plt.imshow(make_mosaic(features[:9], 3, 3, border=1))
        plt.title('%s, FIRST conv layer output' % tracker.feature_type)
    else:
        plt.imshow((tracker.x - tracker.x.min())/(tracker.x.max()-tracker.x.min()))
        plt.title('Feature used is %s' % tracker.feature_type)

    plt.subplot(224)
    if tracker.feature_type == 'multi_cnn':
        plt.imshow(make_mosaic(tracker.response_all[:5], 2, 3, border=5))
    else:
        plt.imshow(tracker.response)
    plt.title('response')
    plt.colorbar()

    plt.draw()
    plt.waitforbuttonpress(0.001)


def show_precision(positions, ground_truth, title):
    """
    Calculates precision for a series of distance thresholds (percentage of
    frames where the distance to the ground truth is within the threshold).
    The results are shown in a new figure.

    Accepts positions and ground truth as Nx2 matrices (for N frames), and
    a title string.
    """
    import pylab
    print("Evaluating tracking results.")
    pylab.ioff()  # interactive mode off
    max_threshold = 50  # used for graphs in the paper

    if positions.shape[0] != ground_truth.shape[0]:
        raise Exception(
            "Could not plot precisions, because the number of ground"
            "truth frames does not match the number of tracked frames.")

    # calculate distances to ground truth over all frames
    delta = positions - ground_truth
    distances = pylab.sqrt((delta[:, 0]**2) + (delta[:, 1]**2))

    # compute precisions
    precisions = pylab.zeros((max_threshold, 1), dtype=float)
    for p in range(max_threshold):
        precisions[p] = pylab.sum(distances <= p, dtype=float) / len(distances)
    print("Representation score at 20 pixels: %f" % precisions[20])

    # plot the precisions
    pylab.figure()  # 'Number', 'off', 'Name',
    pylab.title("Precisions - " + title)
    pylab.plot(precisions, "k-", linewidth=2)
    pylab.xlabel("Threshold")
    pylab.ylabel("Precision")

    pylab.show()
    return precisions


def plot_tracking(frame, img_rgb, tracker):
    from matplotlib.patches import Rectangle
    plt.figure(1)
    plt.clf()

    tracking_figure_axes = plt.subplot(221)
    tracking_rect = Rectangle(
        xy=(tracker.pos[1]-tracker.target_sz[1]/2, tracker.pos[0]-tracker.target_sz[0]/2),
        width=tracker.target_sz[1],
        height=tracker.target_sz[0],
        facecolor='none',
        edgecolor='r',
        )
    tracking_figure_axes.add_patch(tracking_rect)
    plt.imshow(img_rgb)
    plt.title('frame: %d' % frame)

    plt.subplot(222)
    plt.imshow(tracker.im_crop)
    plt.title('previous target pos image')

    plt.subplot(223)
    plt.imshow(tracker.x)
    plt.title('Feature used is %s' % tracker.feature_type)

    plt.subplot(224)
    plt.imshow(tracker.response)
    plt.title('response')
    plt.colorbar()

    plt.draw()
    plt.waitforbuttonpress(0.1)


def load_video_info(video_path):
    """
    Loads all the relevant information for the video in the given path:
    the list of image files (cell array of strings), initial position
    (1x2), target size (1x2), whether to resize the video to half
    (boolean), and the ground truth information for precision calculations
    (Nx2, for N frames). The ordering of coordinates is always [y, x].

    The path to the video is returned, since it may change if the images
    are located in a sub-folder (as is the default for MILTrack's videos).
    """
    import pylab

    # load ground truth from text file (MILTrack's format)
    text_files = glob.glob(os.path.join(video_path, "*_gt.txt"))
    assert text_files, \
        "No initial position and ground truth (*_gt.txt) to load."

    first_file_path = os.path.join(text_files[0])
    #f = open(first_file_path, "r")
    #ground_truth = textscan(f, '%f,%f,%f,%f') # [x, y, width, height]
    #ground_truth = cat(2, ground_truth{:})
    ground_truth = pylab.loadtxt(first_file_path, delimiter=",")
    #f.close()

    # set initial position and size
    first_ground_truth = ground_truth[0, :]
    # target_sz contains height, width
    target_sz = pylab.array([first_ground_truth[3], first_ground_truth[2]])
    # pos contains y, x center
    pos = [first_ground_truth[1], first_ground_truth[0]] \
        + pylab.floor(target_sz / 2)

    #try:
    if True:
        # interpolate missing annotations
        # 4 out of each 5 frames is filled with zeros
        for i in range(4):  # x, y, width, height
            xp = range(0, ground_truth.shape[0], 5)
            fp = ground_truth[xp, i]
            x = range(ground_truth.shape[0])
            ground_truth[:, i] = pylab.interp(x, xp, fp)
        # store positions instead of boxes
        ground_truth = ground_truth[:, [1, 0]] + ground_truth[:, [3, 2]] / 2
    #except Exception as e:
    else:
        print("Failed to gather ground truth data")
        #print("Error", e)
        # ok, wrong format or we just don't have ground truth data.
        ground_truth = []

    # list all frames. first, try MILTrack's format, where the initial and
    # final frame numbers are stored in a text file. if it doesn't work,
    # try to load all png/jpg files in the folder.

    text_files = glob.glob(os.path.join(video_path, "*_frames.txt"))
    if text_files:
        first_file_path = os.path.join(text_files[0])
        #f = open(first_file_path, "r")
        #frames = textscan(f, '%f,%f')
        frames = pylab.loadtxt(first_file_path, delimiter=",", dtype=int)
        #f.close()

        # see if they are in the 'imgs' subfolder or not
        test1_path_to_img = os.path.join(video_path,
                                         "imgs/img%05i.png" % frames[0])
        test2_path_to_img = os.path.join(video_path,
                                         "img%05i.png" % frames[0])
        if os.path.exists(test1_path_to_img):
            video_path = os.path.join(video_path, "imgs/")
        elif os.path.exists(test2_path_to_img):
            video_path = video_path  # no need for change
        else:
            raise Exception("Failed to find the png images")

        # list the files
        img_files = ["img%05i.png" % i
                     for i in range(frames[0], frames[1] + 1)]
        #img_files = num2str((frames{1} : frames{2})', 'img%05i.png')
        #img_files = cellstr(img_files);
    else:
        # no text file, just list all images
        img_files = glob.glob(os.path.join(video_path, "*.png"))
        if len(img_files) == 0:
            img_files = glob.glob(os.path.join(video_path, "*.jpg"))

        assert len(img_files), "Failed to find png or jpg images"

        img_files.sort()

    # if the target is too large, use a lower resolution
    # no need for so much detail
    if pylab.sqrt(pylab.prod(target_sz)) >= 100:
        pos = pylab.floor(pos / 2)
        target_sz = pylab.floor(target_sz / 2)
        resize_image = True
    else:
        resize_image = False

    ret = [img_files, pos, target_sz, resize_image, ground_truth, video_path]
    return ret
