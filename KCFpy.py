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

import matplotlib.pyplot as plt


class KCFTracker:
    def __init__(self, feature_type='raw', debug=False, gt_type='rect'):
        """
        object_example is an image showing the object to track
        feature_type:
            "raw pixels":
            "hog":
            "CNN":
        """
        # parameters according to the paper --
        self.padding = 2.2  # extra area surrounding the target
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
        self.im_sz = []
        self.debug = debug # a flag indicating to plot the intermediate figures

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
        elif self.feature_type == 'vgg':
            from keras.applications.vgg19 import VGG19
            from keras.models import Model

            self.base_model = VGG19(include_top=False, weights='imagenet')
            # self.block4_conv4_model = Model(input=self.base_model.input, output=self.base_model.get_layer('block4_conv4').output)
            # self.block1_conv1_model = Model(input=self.base_model.input, output=self.base_model.get_layer('block1_conv1').output)
            # self.block1_conv2_model = Model(input=self.base_model.input, output=self.base_model.get_layer('block1_conv2').output)

            self.extract_model = Model(input=self.base_model.input, output=self.base_model.get_layer('block3_conv4').output)
            self.cell_size = 4

            self.feature_bandwidth_sigma = 1
            self.adaptation_rate = 0.01

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
        if self.feature_type == 'raw' and im.shape[0] == 3:
            im = im.transpose(1, 2, 0)/255.
            self.im_sz = im.shape
        elif self.feature_type == 'vgg':
            self.im_sz = im.shape[1:]

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
        if self.feature_type == 'raw' and im.shape[0] == 3:
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
        # we also require the bounding box to be within the image boundary
        self.res.append([min(im.shape[0] - self.target_sz[1], max(0, self.pos[1] - self.target_sz[1] / 2.)),
                         min(im.shape[1] - self.target_sz[0], max(0, self.pos[0] - self.target_sz[0] / 2.)),
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
        elif self.feature_type == 'vgg':
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
        if self.feature_type == 'raw':
            im_shape = im.shape
        elif self.feature_type == 'vgg':
            im_shape = im.shape[1:]

        ys[ys < 0] = 0
        ys[ys >= im_shape[0]] = im_shape[0] - 1

        xs[xs < 0] = 0
        xs[xs >= im_shape[1]] = im_shape[1] - 1

        # extract image

        if self.feature_type == 'raw':
            out = im[np.ix_(ys, xs)]
        elif self.feature_type == 'vgg':
            c = np.array(range(3))
            out = im[np.ix_(c, ys, xs)]

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
        elif self.feature_type == 'vgg':
            from keras.applications.vgg19 import preprocess_input
            x = np.expand_dims(self.im_crop.copy(), axis=0)
            x = preprocess_input(x)
            #block4_conv4_features = self.block4_conv4_model.predict(x)
            # block1_conv1_features = self.block1_conv1_model.predict(x)
            # block1_conv1_features = np.squeeze(block1_conv1_features)
            # return block1_conv1_features.transpose(1, 2, 0) / block1_conv1_features.transpose(1, 2, 0).max()

            features = self.extract_model.predict(x)
            features = np.squeeze(features)
            features = features.transpose(1, 2, 0) / features.transpose(1, 2, 0).max()
            features = np.multiply(features, cos_window[:, :, None])

            return features



        else:
            assert 'Non implemented!'


def track(args):
    """
    notation: variables ending with f are in the frequency domain.
    """
    from visualisation_utils import load_video_info, plot_tracking, show_precision
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
