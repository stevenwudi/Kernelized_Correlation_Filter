#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a python reimplementation of the open source tracker in
High-Speed Tracking with Kernelized Correlation Filters
João F. Henriques, Rui Caseiro, Pedro Martins, and Jorge Batista, tPAMI 2015
modified by Di Wu
"""

import os
import glob
import time
import argparse
import cv2
import numpy as np
#from hog import hog
import matplotlib.pyplot as plt


class KCFTracker:
    def __init__(self, feature_type='raw', gt_type='rect'):
        """
        object_example is an image showing the object to track
        feature_type:
            "raw pixels":
            "hog":
            "CNN":
        """
        # parameters according to the paper --
        self.padding = 1.  # extra area surrounding the target
        self.lambda_value = 1e-4  # regularization
        self.spatial_bandwidth_sigma_factor = 1 / float(16)
        self.feature_type = feature_type
        self.patch_size = []
        self.output_sigma = []
        self.cos_window = []
        self.pos = []
        self.x = []
        self.alphaf = []
        self.xf = []
        self.yf = []
        self.im_crop = []
        self.response = []
        self.target_out = []
        self.target_sz = []
        self.vert_delta = 0
        self.horiz_delta = 0
        # OBT dataset need extra definition
        self.name = 'KCF' + feature_type
        self.fps = -1
        self.type = gt_type
        self.res = []

        # following is set according to Table 2:
        if self.feature_type == 'raw':
            self.adaptation_rate = 0.075  # linear interpolation factor for adaptation
            self.feature_bandwidth_sigma = 0.2
            self.cell_size = 1
        elif self.feature_type == 'hog':
            self.adaptation_rate = 0.02  # linear interpolation factor for adaptation
            self.bin_num = 31
            self.cell_size = 4
            self.feature_bandwidth_sigma = 0.5

    def train(self, im, init_rect, target_sz):
        """
        :param im: image should be of 3 dimension: M*N*C
        :param pos: the centre position of the target
        :param target_sz: target size
        """
        self.pos = [init_rect[1]+init_rect[3]/2., init_rect[0]+init_rect[2]/2.]
        self.res.append(init_rect)
        # Duh OBT is the reverse
        self.target_sz = target_sz[::-1]
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        # desired output (gaussian shaped), bandwidth proportional to target size
        self.output_sigma = np.sqrt(np.prod(self.target_sz)) * self.spatial_bandwidth_sigma_factor
        grid_y = np.arange(np.floor(self.patch_size[0]/self.cell_size)) - np.floor(self.patch_size[0]/(2*self.cell_size))
        grid_x = np.arange(np.floor(self.patch_size[1]/self.cell_size)) - np.floor(self.patch_size[1]/(2*self.cell_size))
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.yf = self.fft2(y)

        # store pre-computed cosine window
        self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))

        # extract and pre-process subwindow
        if im.shape[0] == 3:
            im = im.transpose(1, 2, 0)/255.

        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        self.x = self.get_features(cos_window=self.cos_window)
        self.xf = self.fft2(self.x)
        k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x)
        self.alphaf = np.divide(self.yf, self.fft2(k) + self.lambda_value)
        self.response = np.real(np.fft.ifft2(np.multiply(self.alphaf, self.fft2(k))))

    def detect(self, im):
        """
        Note: we assume the target does not change in scale, hence there is no target size

        :param im: image should be of 3 dimension: M*N*C
        :return:
        """

        # extract and pre-process subwindow
        if im.shape[0]==3:
            im = im.transpose(1, 2, 0)/255.
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        z = self.get_features(cos_window=self.cos_window)
        zf = self.fft2(z)
        k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x, zf, z)
        kf = self.fft2(k)
        self.response = np.real(np.fft.ifft2(np.multiply(self.alphaf, kf)))

        # target location is at the maximum response. We must take into account the fact that, if
        # the target doesn't move, the peak will appear at the top-left corner, not at the centre
        # (this is discussed in the paper Fig. 6). The response map wrap around cyclically.
        v_centre, h_centre = np.unravel_index(self.response.argmax(), self.response.shape)
        self.vert_delta, self.horiz_delta = [v_centre - self.response.shape[0]/2, h_centre - self.response.shape[1]/2]
        self.pos = self.pos + np.dot(self.cell_size, [self.vert_delta, self.horiz_delta])
        self.res.append([self.pos[1] - self.target_sz[1] / 2., self.pos[0] - self.target_sz[0] / 2,
                         self.target_sz[1], self.target_sz[0]])
        #########################################
        # we need to train the tracker again here, it's almost the replicate of train
        #########################################
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        x_new = self.get_features(cos_window=self.cos_window)
        xf_new = self.fft2(x_new)
        k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new, x_new)
        kf = self.fft2(k)
        alphaf_new = np.divide(self.yf, kf + self.lambda_value)

        self.x = (1 - self.adaptation_rate) * self.x + self.adaptation_rate * x_new
        self.xf = (1 - self.adaptation_rate) * self.xf + self.adaptation_rate * xf_new
        self.alphaf = (1-self.adaptation_rate) * self.alphaf + self.adaptation_rate * alphaf_new

        return self.pos

    def dense_gauss_kernel(self, sigma, xf, x, zf=None, z=None):
        """
        Gaussian Kernel with dense sampling.
        Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
        between input images X and Y, which must both be MxN. They must also
        be periodic (ie., pre-processed with a cosine window). The result is
        an MxN map of responses.

        If X and Y are the same, ommit the third parameter to re-use some
        values, which is faster.
        :param sigma: feature bandwidth sigma
        :param x:
        :param y: if y is None, then we calculate the auto-correlation
        :return:
        """
        N = xf.shape[0]*xf.shape[1]
        xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

        if zf is None:
            # auto-correlation of x
            zf = xf
            zz = xx
        else:
            zz = np.dot(z.flatten().transpose(), z.flatten())  # squared norm of y

        xyf = np.multiply(zf, np.conj(xf))
        if self.feature_type == 'raw':
            xyf_ifft = np.fft.ifft2(xyf)
        elif self.feature_type == 'hog':
            xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))

        row_shift, col_shift = np.floor(np.array(xyf_ifft.shape) / 2).astype(int)
        xy_complex = np.roll(xyf_ifft, row_shift, axis=0)
        xy_complex = np.roll(xy_complex, col_shift, axis=1)
        c = np.real(xy_complex)
        d = np.real(xx) + np.real(zz) - 2 * c
        k = np.exp(-1 / sigma**2 * np.maximum(0, d) / N)

        return k

    def get_subwindow(self, im, pos, sz):
        """
        Obtain sub-window from image, with replication-padding.
        Returns sub-window of image IM centered at POS ([y, x] coordinates),
        with size SZ ([height, width]). If any pixels are outside of the image,
        they will replicate the values at the borders.

        The subwindow is also normalized to range -0.5 .. 0.5, and the given
        cosine window COS_WINDOW is applied
        (though this part could be omitted to make the function more general).
        """

        if np.isscalar(sz):  # square sub-window
            sz = [sz, sz]

        ys = np.floor(pos[0]) + np.arange(sz[0], dtype=int) - np.floor(sz[0] / 2)
        xs = np.floor(pos[1]) + np.arange(sz[1], dtype=int) - np.floor(sz[1] / 2)

        ys = ys.astype(int)
        xs = xs.astype(int)

        # check for out-of-bounds coordinates,
        # and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= im.shape[0]] = im.shape[0] - 1

        xs[xs < 0] = 0
        xs[xs >= im.shape[1]] = im.shape[1] - 1

        # extract image
        out = im[np.ix_(ys, xs)]
        return out

    def fft2(self, x):
        """
        FFT transform of the first 2 dimension
        :param x: M*N*C the first two dimensions are used for Fast Fourier Transform
        :return:  M*N*C the FFT2 of the first two dimension
        """
        return np.fft.fft2(x, axes=(0, 1))


    def get_features(self, cos_window):
        """

        :param cos_window:
        :return:
        """
        if self.feature_type == 'raw':
            # using only grayscale:
            if len(self.im_crop.shape) == 3:
                img_gray = np.mean(self.im_crop, axis=2)
                img_gray = img_gray - img_gray.mean()
                features = np.multiply(img_gray, cos_window)
                return features

        elif self.feature_type == 'hog':
            # using only grayscale:
            if len(self.im_crop.shape) == 3:
                img_gray = cv2.cvtColor(self.im_crop, cv2.COLOR_BGR2GRAY)
                features_hog, hog_image = hog(img_gray, orientations=self.bin_num, pixels_per_cell=(self.cell_size, self.cell_size), cells_per_block=(1, 1), visualise=True)
                features = np.multiply(features_hog, cos_window[:, :, None])
                return features
        else:
            assert 'Non implemented!'


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
    plt.waitforbuttonpress(1)


def plot_tracking_rect(frame, img_rgb, tracker):
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
    plt.waitforbuttonpress(0.01)


def track(args):
    """
    notation: variables ending with f are in the frequency domain.
    """

    info = load_video_info(args.video_path)
    img_files, pos, target_sz, should_resize_image, ground_truth, video_path = info
    tracker = KCFTracker()

    positions = np.zeros((len(img_files), 2))  # to calculate precision
    total_time = 0
    start_time = time.time()
    for frame, image_filename in enumerate(img_files):
        image_path = os.path.join(video_path, image_filename)
        img_rgb = cv2.imread(image_path)

        if frame == 0:
            tracker.train(img_rgb, pos, target_sz)
        else:
            pos = tracker.detect(img_rgb)
            positions[frame, :] = pos

        print("Frame ==", frame)
        print('vert_delta: %.2f, horiz_delta: %.2f' % (tracker.vert_delta, tracker.horiz_delta))
        print("pos", pos)
        print("gt", ground_truth[frame])
        print("\n")

        args.debug = False
        if args.debug:
            plot_tracking(frame, img_rgb, tracker)

    total_time += time.time() - start_time
    print("Frames-per-second:",  len(img_files) / total_time)

    title = os.path.basename(os.path.normpath(args.video_path))
    if len(ground_truth) > 0:
        precisions = show_precision(positions, ground_truth, title)


    # spatial_bandwidth_sigma_factor
    # 1/float(10): Representation score at 20 pixels: 0.386301
    # 1/float(16): Representation score at 20 pixels: 0.641096
    # What the fuck!

    # lambda:
    # 1e-4: Representation score at 20 pixels: 0.641096
    # 1e-2: Representation score at 20 pixels: 0.583562

    # Representation score at 20 pixels: 0.621918 (Old one)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="debug mode, plotting some intermediate figures",
                        default=True, action="store_true")
    parser.add_argument("-v", "--video_path", help="video path",
                        default="./data/tiger2")
    args = parser.parse_args()
    track(args)


if __name__ == "__main__":
    main()
