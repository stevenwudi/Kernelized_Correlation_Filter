import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import cv2


class KCFTracker:
    def __init__(self,
                 feature_type='HDT',
                 padding=2.2,
                 feature_bandwidth_sigma=0.2,
                 spatial_bandwidth_sigma_factor=1 / float(16),
                 adaptation_rate_range_max=0.0025,
                 adaptation_rate_scale_range_max=0.005,
                 lambda_value=1e-4,
                 kernel='gaussian',
                 sub_feature_type="",
                 sub_sub_feature_type=""
                 ):
        """
        :param feature_type:
        :param padding:
        :param feature_bandwidth_sigma:
        :param spatial_bandwidth_sigma_factor:
        :param adaptation_rate_range_max:
        :param adaptation_rate_scale_range_max:
        :param lambda_value:
        :param kernel:
        """
        self.feature_type = feature_type
        self.padding = padding
        self.feature_bandwidth_sigma = feature_bandwidth_sigma
        self.spatial_bandwidth_sigma_factor = spatial_bandwidth_sigma_factor
        self.adaptation_rate_range_max = adaptation_rate_range_max
        self.adaptation_rate_scale_range_max = adaptation_rate_scale_range_max
        self.lambda_value = lambda_value
        self.kernel = kernel
        self.sub_sub_feature_type = sub_sub_feature_type
        self.name = 'KCF_' + feature_type

        if feature_type == 'HDT':
            from keras.applications.vgg19 import VGG19
            import theano

            self.base_model = VGG19(include_top=False, weights='imagenet')
            self.extract_model_function = theano.function([self.base_model.input],
                                                          [self.base_model.get_layer('block1_conv2').output,
                                                           self.base_model.get_layer('block2_conv2').output,
                                                           self.base_model.get_layer('block3_conv4').output,
                                                           self.base_model.get_layer('block4_conv4').output,
                                                           self.base_model.get_layer('block5_conv4').output
                                                           ], allow_input_downcast=True)

            # we first resize all the response maps to a size of 40*60 (store the resize scale)
            # because average target size is 81 *52
            self.resize_size = (241, 161)
            # store pre-computed cosine window, here is a multiscale CNN, here we have 5 layers cnn:
            self.cos_window = []
            self.y = []
            self.yf = []
            self.response_all = []
            self.max_list = []
            for i in range(5):
                cos_wind_sz = np.divide(self.resize_size, 2**i)
                self.cos_window.append(np.outer(np.hanning(cos_wind_sz[0]), np.hanning(cos_wind_sz[1])))
                grid_y = np.arange(cos_wind_sz[0]) - np.floor(cos_wind_sz[0] / 2)
                grid_x = np.arange(cos_wind_sz[1]) - np.floor(cos_wind_sz[1] / 2)
                # desired output (gaussian shaped), bandwidth proportional to target size
                output_sigma = np.sqrt(np.prod(cos_wind_sz)) * self.spatial_bandwidth_sigma_factor
                rs, cs = np.meshgrid(grid_x, grid_y)
                y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
                self.y.append(y)
                self.yf.append(np.fft.fft2(y, axes=(0, 1)))

            # some hyperparameter for HDT
            self.loss_acc_time = 5
            self.stability = np.zeros(shape=(5, 1))
            self.A = 0.011

    def train(self, im, init_rect, seqname=""):
        """
        The function for train the first frame
        :param im:
        :param init_rect:
        :param seqnem:
        :return:
        """
        self.pos = [init_rect[1]+init_rect[3]/2., init_rect[0]+init_rect[2]/2.]
        self.res = []
        self.res.append(init_rect)
        # for scaling, we always need to set it to 1
        self.currentScaleFactor = 1
        # Duh OBT is the reverse
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        # first_target_sz is used to save the first frame size for the scale changes in the detection
        self.first_target_sz = self.target_sz
        self.first_patch_sz = np.array(self.patch_size).astype(int)
        # desired output (gaussian shaped), bandwidth proportional to target size
        self.output_sigma = np.sqrt(np.prod(self.target_sz)) * self.spatial_bandwidth_sigma_factor
        self.im_sz = im.shape[1:]

        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        self.x = self.get_features()
        self.xf = self.fft2(self.x)
        self.alphaf = []

        if self.feature_type == 'HDT':
            # store pre-computed cosine window, here is a multiscale CNN, here we have 5 layers cnn:
            #self.W = np.asarray([0.05, 0.1, 0.2, 0.5, 1])
            self.W = np.ones(5) * (1/5.)
            self.W = self.W / np.sum(self.W)
            self.R = np.zeros(shape=(len(self.W)))
            self.loss = np.zeros(shape=(self.loss_acc_time + 1, len(self.W)))

            for l in range(len(self.x)):
                if self.kernel == 'gaussian':
                    k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[l], self.x[l])
                    kf = np.fft.fft2(k, axes=(0, 1))
                elif self.kernel == 'linear':
                    kf = self.linear_kernel(self.xf[l])
                self.alphaf.append(np.divide(self.yf[l], kf + self.lambda_value))

    def detect(self, im, frame):
        """
        :param im:
        :param frame:
        :return:
        """
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        z = self.get_features()
        zf = self.fft2(z)

        if self.feature_type == 'HDT':
            self.response = []
            for i in range(len(z)):
                if self.kernel == 'gaussian':
                    k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i], zf[i], z[i])
                    kf = self.fft2(k)
                elif self.kernel == 'linear':
                    kf = self.linear_kernel(self.xf[i], zf[i])
                self.response.append(np.real(np.fft.ifft2(np.multiply(self.alphaf[i], kf))))

            self.max_list = np.asarray([np.max(x) for x in self.response])
            self.maxres = np.zeros(shape=len(self.response))
            self.expert_row = np.zeros(shape=len(self.response))
            self.expert_col = np.zeros(shape=len(self.response))
            self.response_all = np.zeros(shape=(len(self.response), self.resize_size[0], self.resize_size[1]))
            self.cell_size_all = np.zeros(shape=(len(self.response), 2))

            row = 0
            col = 0
            for l in range(len(self.response)):
                rm = self.response[l]
                # we reshape the response to the same size (for visualisation process)
                rm_resize = imresize(rm, self.resize_size)
                self.response_all[l] = rm_resize
                self.maxres[l] = rm.max()
                self.expert_row[l], self.expert_col[l] = np.unravel_index(rm.argmax(), rm.shape)
                self.cell_size_all[l] = np.divide(self.patch_size, rm.shape)
                row += self.W[l] * self.expert_row[l] * self.cell_size_all[l][0]
                col += self.W[l] * self.expert_col[l] * self.cell_size_all[l][1]

            self.vert_delta, self.horiz_delta = [row - self.patch_size[0] / 2.,
                                                 col - self.patch_size[1] / 2.]
            self.pos = [self.pos[0] + self.vert_delta - 1, self.pos[1] + self.horiz_delta - 1]

            for l in range(len(self.response)):
                self.loss[-1, l] = self.maxres[l] - \
                                    self.response[l][
                                        np.rint(row / self.cell_size_all[l][0]), np.rint(col / self.cell_size_all[l][1])]
                # update the loss history
            self.LosIdx = np.mod(frame - 1, self.loss_acc_time)
            self.loss[self.LosIdx] = self.loss[-1]

            if frame > self.loss_acc_time:
                self.lossA = np.sum(np.multiply(self.W, self.loss[-1]))
                self.LosMean = np.mean(self.loss[:self.loss_acc_time], axis=0)
                self.LosStd = np.std(self.loss[:self.loss_acc_time], axis=0)
                self.LosMean[self.LosMean < 0.0001] = 0
                self.LosStd[self.LosStd < 0.0001] = 0

                self.curDiff = self.loss[-1] - self.LosMean
                self.stability = np.exp(( -1 * np.divide(np.abs(self.curDiff), self.LosStd + np.finfo(float).eps)))
                print("stability is {0}".format(self.stability))
                self.alpha = np.clip(self.stability, 0.12, 0.97)
                self.R = np.multiply(self.R, self.alpha) + np.multiply((1 - self.alpha), self.lossA - self.loss[-1])
                print("Regret is {0}".format(self.R))
                self.c = self.find_nh_scale(self.R, self.A)
                self.W = self.nnhedge_weights(self.R, self.c, self.A)

                #self.W = np.multiply(self.W, self.max_list)
                #self.W = np.multiply(self.W, self.stability)
                self.W = np.clip(self.W / np.sum(self.W), 0.001, 1)
                self.W = self.W / np.sum(self.W)
            print("W is {0}".format(self.W))

        ##################################################################################
        # we need to train the tracker again here, it's almost the replicate of train
        ##################################################################################
        # we update the model from here
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        x_new = self.get_features()

        if self.feature_type == 'HDT':
            xf_new = self.fft2(x_new)
            # Wudi's new invention for adaptive learning rate
            adaptation_rate = self.stability * self.adaptation_rate_range_max
            #adaptation_rate = np.ones(shape=(self.stability.shape)) * self.adaptation_rate_range_max
            for i in range(len(x_new)):
                if self.kernel == 'gaussian':
                    k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new[i], x_new[i])
                    kf = np.fft.fft2(k, axes=(0, 1))
                elif self.kernel == 'linear':
                    kf = self.linear_kernel(xf_new[i])
                alphaf_new = np.divide(self.yf[i], kf + self.lambda_value)
                self.x[i] = (1 - adaptation_rate[i]) * self.x[i] + adaptation_rate[i] * x_new[i]
                self.xf[i] = (1 - adaptation_rate[i]) * self.xf[i] + adaptation_rate[i] * xf_new[i]
                self.alphaf[i] = (1 - adaptation_rate[i]) * self.alphaf[i] + adaptation_rate[i] * alphaf_new

        # we also require the bounding box to be within the image boundary
        self.res.append([min(self.im_sz[1] - self.target_sz[1], max(0, self.pos[1] - self.target_sz[1] / 2.)),
                         min(self.im_sz[0] - self.target_sz[0], max(0, self.pos[0] - self.target_sz[0] / 2.)),
                         self.target_sz[1], self.target_sz[0]])

        return self.pos

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

        # check for out-of-bounds coordinates and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= self.im_sz[0]] = self.im_sz[0] - 1

        xs[xs < 0] = 0
        xs[xs >= self.im_sz[1]] = self.im_sz[1] - 1

        # extract image
        if self.feature_type == 'HDT':
            c = np.array(range(3))
            out = im[np.ix_(c, ys, xs)]
            # if self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn':
            #     from keras.applications.vgg19 import preprocess_input
            #     x = imresize(out.copy(), self.resize_size)
            #     out = np.multiply(x, self.cos_window_patch[:, :, None])
            return out

    def get_features(self):
        """
        :param im: input image
        :return:
        """
        if self.feature_type == "HDT":
            from keras.applications.vgg19 import preprocess_input
            x = imresize(self.im_crop.copy(), self.resize_size)
            x = x.transpose((2, 0, 1)).astype(np.float64)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features_list = self.extract_model_function(x)
            for i, features in enumerate(features_list):
                features = np.squeeze(features)
                features = (features.transpose(1, 2, 0) - features.min()) / (features.max() - features.min())
                features_list[i] = np.multiply(features, self.cos_window[i][:, :, None])
            return features_list
        else:
            assert 'Non implemented!'

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
        N = xf.shape[0] * xf.shape[1]
        xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x

        if zf is None:
            # auto-correlation of x
            zf = xf
            zz = xx
        else:
            zz = np.dot(z.flatten().transpose(), z.flatten())  # squared norm of y

        xyf = np.multiply(zf, np.conj(xf))
        if self.feature_type == 'HDT':
            xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))

        row_shift, col_shift = np.floor(np.array(xyf_ifft.shape) / 2).astype(int)
        xy_complex = np.roll(xyf_ifft, row_shift, axis=0)
        xy_complex = np.roll(xy_complex, col_shift, axis=1)
        c = np.real(xy_complex)
        d = np.real(xx) + np.real(zz) - 2 * c

        D = np.maximum(0, d) / N
        #sigma = np.sum(np.diag(np.cov(D)))
        k = np.exp(-1. / sigma ** 2 * D)

        return k

    def linear_kernel(self, xf, zf=None):
        if zf is None:
            zf = xf
        N = np.prod(xf.shape)
        xyf = np.multiply(zf, np.conj(xf))
        kf = np.sum(xyf, axis=2)
        return kf / N

    def fft2(self, x):
        """
        FFT transform of the first 2 dimension
        :param x: M*N*C the first two dimensions are used for Fast Fourier Transform
        :return:  M*N*C the FFT2 of the first two dimension
        """
        if type(x) == list:
            x = [np.fft.fft2(f, axes=(0, 1)) for f in x]
            return x
        else:
            return np.fft.fft2(x, axes=(0, 1))

    def find_nh_scale(self, regrets, A):

        def avgnh(r, c, A):
            n = np.prod(r.shape)
            T = r + A
            T[T<0] = 0
            w = np.exp(0.5 * np.multiply(T, T) / c)
            total = (1./n) * np.sum(w) - 2.72
            return total

        # first find an upper and lower bound on c, based on the nh weights
        clower = 1
        counter = 0
        while avgnh(regrets, clower, A) < 0 and counter < 30:
            clower *= 0.5
            counter += 1

        cupper = 1
        counter = 0
        while avgnh(regrets, cupper, A) > 0 and counter < 30:
            cupper *= 2
            counter += 1

        # now dow a binary search
        cmid = (cupper + clower) /2
        counter = 0
        while np.abs(avgnh(regrets, cmid, A))> 1e-2 and counter < 30:
            if avgnh(regrets, cmid, A) > 1e-2:
                clower = cmid
                cmid = (cmid + cupper) / 2
            else:
                cupper = cmid
                cmid = (cmid + clower) / 2
            counter += 1

        return cmid

    def nnhedge_weights(self, r, scale, A):
        n = np.prod(r.shape)
        w = np.zeros(shape=n)

        for i in range(n):
            if r[i] + A <= 0:
                w[i] = 0
            else:
                w[i] = (r[i] + A)/scale * np.exp((r[i] + A) * (r[i] + A) / (2 * scale))
        return w