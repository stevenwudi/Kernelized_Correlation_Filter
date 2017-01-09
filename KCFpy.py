"""
This is a python reimplementation of the open source tracker in
High-Speed Tracking with Kernelized Correlation Filters
Joao F. Henriques, Rui Caseiro, Pedro Martins, and Jorge Batista, tPAMI 2015
modified by Di Wu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize


class KCFTracker:
    def __init__(self, feature_type='raw', sub_feature_type='', debug=False, gt_type='rect', load_model=False):
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
        self.sub_feature_type = sub_feature_type
        self.name = 'KCF' + feature_type
        if self.sub_feature_type:
            self.name += '_'+sub_feature_type
        self.fps = -1
        self.type = gt_type
        self.res = []
        self.im_sz = []
        self.debug = debug # a flag indicating to plot the intermediate figures
        self.first_patch_sz = []
        self.first_target_sz = []
        self.currentScaleFactor = 1
        self.load_model = load_model

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
        elif self.feature_type == 'dsst':
            # this method adopts from the paper  Martin Danelljan, Gustav Hger, Fahad Shahbaz Khan and Michael Felsberg.
            # "Accurate Scale Estimation for Robust Visual Tracking". (BMVC), 2014.
            # The project website is: http: // www.cvl.isy.liu.se / research / objrec / visualtracking / index.html
            self.adaptation_rate = 0.025  # linear interpolation factor for adaptation
            self.feature_bandwidth_sigma = 0.2
            self.cell_size = 1
            self.scale_step = 1.02
            self.nScales = 33
            self.scaleFactors = self.scale_step **(np.ceil(self.nScales * 1.0/ 2) - range(1, self.nScales+1))
            self.scale_window = np.hanning(self.nScales)
            self.scale_sigma_factor = 1./4
            self.scale_sigma = self.nScales / np.sqrt(self.nScales) * self.scale_sigma_factor
            self.ys = np.exp(-0.5 * ((range(1, self.nScales+1) - np.ceil(self.nScales * 1.0 /2))**2) / self.scale_sigma**2)
            self.ysf = np.fft.fft(self.ys)
            self.min_scale_factor = []
            self.max_scale_factor = []
            self.xs = []
            self.xsf = []
            # we use linear kernel as in the BMVC2014 paper
            self.new_sf_num = []
            self.new_sf_den = []
            self.scale_response = []
            self.lambda_scale = 1e-2
        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50':
            if self.feature_type == 'vgg':
                from keras.applications.vgg19 import VGG19
                from keras.models import Model
                self.base_model = VGG19(include_top=False, weights='imagenet')
                self.extract_model = Model(input=self.base_model.input, output=self.base_model.get_layer('block2_conv2').output)
            elif self.feature_type == 'resnet50':
                from keras.applications.resnet50 import ResNet50
                from keras.models import Model
                self.base_model = ResNet50(weights='imagenet', include_top=False)
                self.extract_model = Model(input=self.base_model.input,
                                           output=self.base_model.get_layer('activation_10').output)
            self.cell_size = 2
            self.feature_bandwidth_sigma = 1
            self.adaptation_rate = 0.01
        elif self.feature_type == 'vgg_rnn':
            from keras.applications.vgg19 import VGG19
            from keras.models import Model
            self.base_model = VGG19(include_top=False, weights='imagenet')
            self.extract_model = Model(input=self.base_model.input,
                                       output=self.base_model.get_layer('block3_conv4').output)
            # we first resize the response map to a size of 50*80 (store the resize scale)
            # because average target size is 81 *52
            self.resize_size = (240, 160)
            self.cell_size = 4
            self.response_size = [self.resize_size[0] / self.cell_size,
                                  self.resize_size[1] / self.cell_size]
            self.feature_bandwidth_sigma = 10
            self.adaptation_rate = 0.01

            grid_y = np.arange(self.response_size[0]) - np.floor(self.response_size[0] / 2)
            grid_x = np.arange(self.response_size[1]) - np.floor(self.response_size[1] / 2)

            # desired output (gaussian shaped), bandwidth proportional to target size
            self.output_sigma = np.sqrt(np.prod(self.response_size)) * self.spatial_bandwidth_sigma_factor
            rs, cs = np.meshgrid(grid_x, grid_y)
            self.y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
            self.yf = self.fft2(self.y)
            # store pre-computed cosine window
            self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))
            self.path_resize_size = np.multiply(self.yf.shape, (1 + self.padding))
            self.cos_window_patch = np.outer(np.hanning(self.resize_size[0]), np.hanning(self.resize_size[1]))
            # Embedding
            if load_model:
                from keras.models import load_model
                self.lstm_model = load_model('rnn_translation_no_scale_freezconv.h5')
                self.lstm_input = np.zeros(shape=(1,10,1,60,40)).astype(float)
        elif self.feature_type == 'cnn':
            from keras.applications.vgg19 import VGG19
            from keras.models import Model
            self.base_model = VGG19(include_top=False, weights='imagenet')
            self.extract_model = Model(input=self.base_model.input,
                                       output=self.base_model.get_layer('block3_conv4').output)
            # we first resize the response map to a size of 50*80 (store the resize scale)
            # because average target size is 81 *52
            self.resize_size = (240, 160)
            self.cell_size = 4
            self.response_size = [self.resize_size[0] / self.cell_size,
                                  self.resize_size[1] / self.cell_size]
            self.feature_bandwidth_sigma = 10
            self.adaptation_rate = 0.01

            grid_y = np.arange(self.response_size[0]) - np.floor(self.response_size[0] / 2)
            grid_x = np.arange(self.response_size[1]) - np.floor(self.response_size[1] / 2)

            # desired output (gaussian shaped), bandwidth proportional to target size
            self.output_sigma = np.sqrt(np.prod(self.response_size)) * self.spatial_bandwidth_sigma_factor
            rs, cs = np.meshgrid(grid_x, grid_y)
            y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
            self.yf = self.fft2(y)
            # store pre-computed cosine window
            self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))
            self.path_resize_size = np.multiply(self.yf.shape, (1 + self.padding))
            self.cos_window_patch = np.outer(np.hanning(self.resize_size[0]), np.hanning(self.resize_size[1]))
            # Embedding
            if load_model:
                from keras.models import load_model
                self.cnn_model = load_model('cnn_translation_scale_combine.h5')
        elif self.feature_type == 'multi_cnn':
            from keras.applications.vgg19 import VGG19
            from keras.models import Model
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
            self.resize_size = (240, 160)
            self.cell_size = 4
            self.response_size = [self.resize_size[0] / self.cell_size,
                                  self.resize_size[1] / self.cell_size]
            self.feature_bandwidth_sigma = 0.2
            self.adaptation_rate = 0.01
            # store pre-computed cosine window, here is a multiscale CNN, here we have 5 layers cnn:
            self.cos_window = []
            self.y = []
            self.yf = []
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
                self.yf.append(self.fft2(y))

            # self.path_resize_size = np.multiply(self.yf.shape, (1 + self.padding))
            # self.cos_window_patch = np.outer(np.hanning(self.resize_size[0]), np.hanning(self.resize_size[1]))
            # Embedding
            if load_model:
                from keras.models import load_model
                self.multi_cnn_model = load_model('./models/CNN_Model_OBT100_multi_cnn_final.h5')

    def train(self, im, init_rect):
        """
        :param im: image should be of 3 dimension: M*N*C
        :param pos: the centre position of the target
        :param target_sz: target size
        """
        self.pos = [init_rect[1]+init_rect[3]/2., init_rect[0]+init_rect[2]/2.]
        self.res.append(init_rect)
        # Duh OBT is the reverse
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.first_target_sz = self.target_sz  # because we might introduce the scale changes in the detection
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.first_patch_sz = np.array(self.patch_size).astype(int)  # because we might introduce the scale changes in the detection
        # desired output (gaussian shaped), bandwidth proportional to target size
        self.output_sigma = np.sqrt(np.prod(self.target_sz)) * self.spatial_bandwidth_sigma_factor
        grid_y = np.arange(np.floor(self.patch_size[0]/self.cell_size)) - np.floor(self.patch_size[0]/(2*self.cell_size))
        grid_x = np.arange(np.floor(self.patch_size[1]/self.cell_size)) - np.floor(self.patch_size[1]/(2*self.cell_size))
        if self.feature_type == 'resnet50':
            # this is an odd tweak to make the dimension uniform:
            if np.mod(self.patch_size[0], 2) == 0:
                grid_y = np.arange(np.floor(self.patch_size[0] / self.cell_size)-1) - np.floor(
                    self.patch_size[0] / (2 * self.cell_size)) - 0.5
            if np.mod(self.patch_size[1], 2) == 0:
                grid_x = np.arange(np.floor(self.patch_size[1] / self.cell_size)-1) - np.floor(
                    self.patch_size[1] / (2 * self.cell_size)) - 0.5

        if self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn':
            grid_y = np.arange(self.response_size[0]) - np.floor(self.response_size[0]/2)
            grid_x = np.arange(self.response_size[1]) - np.floor(self.response_size[1]/2)

        if not self.feature_type == 'multi_cnn':
            rs, cs = np.meshgrid(grid_x, grid_y)
            self.y = np.exp(-0.5 / self.output_sigma ** 2 * (rs ** 2 + cs ** 2))
            self.yf = self.fft2(self.y)
            # store pre-computed cosine window
            self.cos_window = np.outer(np.hanning(self.yf.shape[0]), np.hanning(self.yf.shape[1]))

        # extract and pre-process subwindow
        if self.feature_type == 'raw' and im.shape[0] == 3:
            im = im.transpose(1, 2, 0)/255.
            self.im_sz = im.shape
        elif self.feature_type == 'dsst':
            im = im.transpose(1, 2, 0) / 255.
            self.im_sz = im.shape
            self.min_scale_factor = self.scale_step **(np.ceil(np.log(max(5. / self.patch_size)) / np.log(self.scale_step)))
            self.max_scale_factor = self.scale_step **(np.log(min(np.array(self.im_sz[:2]).astype(float) / self.target_sz)) / np.log(self.scale_step))

            self.xs = self.get_scale_sample(im, self.currentScaleFactor * self.scaleFactors)
            self.xsf = np.fft.fftn(self.xs, axes=[0])
            # we use linear kernel as in the BMVC2014 paper
            self.new_sf_num = np.multiply(self.ysf[:, None], np.conj(self.xsf))
            self.new_sf_den = np.real(np.sum(np.multiply(self.xsf, np.conj(self.xsf)), axis=1))

        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50' or \
                        self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn' or self.feature_type == 'multi_cnn':
            self.im_sz = im.shape[1:]

        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)

        self.x = self.get_features()
        self.xf = self.fft2(self.x)
        if self.feature_type == 'cnn' or self.feature_type == 'vgg':
            corr = np.multiply(self.x, self.y[:, :, None])
            corr = np.sum(np.sum(corr, axis=0), axis=0)
            # we compute the correlation of a filter within a layer to its features
            self.feature_correlation = (corr-corr.min())/(corr.max()-corr.min())

        if self.feature_type == 'multi_cnn':
            # multi_cnn will render the models to be of a list
            self.alphaf = []
            for i in range(len(self.x)):
                k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i])
                self.alphaf.append(np.divide(self.yf[i], self.fft2(k) + self.lambda_value))
        else:
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x)
            self.alphaf = np.divide(self.yf, self.fft2(k) + self.lambda_value)
            self.response = np.real(np.fft.ifft2(np.multiply(self.alphaf, self.fft2(k))))

        # the first frame also need to be included!
        self.res.append([min(self.im_sz[1] - self.target_sz[1], max(0, self.pos[1] - self.target_sz[1] / 2.)),
                         min(self.im_sz[0] - self.target_sz[0], max(0, self.pos[0] - self.target_sz[0] / 2.)),
                         self.target_sz[1], self.target_sz[0]])

    def detect(self, im, frame):
        """
        Note: we assume the target does not change in scale, hence there is no target size
        :param im: image should be of 3 dimension: M*N*C
        :return:
        """
        # extract and pre-process subwindow
        if self.feature_type == 'raw' and im.shape[0] == 3:
            im = im.transpose(1, 2, 0)/255.

        elif self.feature_type == 'dsst':
            im = im.transpose(1, 2, 0) / 255.
            self.im_sz = im.shape

        # Qutoe from BMVC2014paper: Danelljan:
        # "In visual tracking scenarios, the scale difference between two frames is typically smaller compared to the
        # translation. Therefore, we first apply the translation filter hf given a new frame, afterwards the scale
        # filter hs is applied at the new target location.
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        z = self.get_features()
        zf = self.fft2(z)

        if not self.feature_type == 'multi_cnn':
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x, zf, z)
            kf = self.fft2(k)
            self.response = np.real(np.fft.ifft2(np.multiply(self.alphaf, kf)))
        else:
            self.response = []
            for i in range(len(z)):
                k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i], zf[i], z[i])
                kf = self.fft2(k)
                self.response.append(np.real(np.fft.ifft2(np.multiply(self.alphaf[i], kf))))

        if self.feature_type == 'raw' or self.feature_type == 'vgg':
            # target location is at the maximum response. We must take into account the fact that, if
            # the target doesn't move, the peak will appear at the top-left corner, not at the centre
            # (this is discussed in the paper Fig. 6). The response map wrap around cyclically.
            v_centre, h_centre = np.unravel_index(self.response.argmax(), self.response.shape)
            self.vert_delta, self.horiz_delta = [v_centre - self.response.shape[0] / 2,
                                                 h_centre - self.response.shape[1] / 2]
            self.pos = self.pos + np.dot(self.cell_size, [self.vert_delta, self.horiz_delta])
        elif self.feature_type == 'vgg_rnn':
            # We need to normalise it (because our training did so):
            response = self.response
            response = (response - response.min()) / (response.max() - response.min())
            response = np.expand_dims(np.expand_dims(response, axis=0), axis=0)

            if frame <= 10:
                self.lstm_input[0, frame-1, :, :, :] = response
                predicted_output_all = self.lstm_model.predict(self.lstm_input, batch_size=1)
                predicted_output = predicted_output_all[0, frame-1,:2]
            else:
                # we always shift the frame to the left and have the final output prediction
                self.lstm_input[0, 0:9, :, :, :] = self.lstm_input[0, 1:10, :, :, :]
                self.lstm_input[0, 9, :, :, :] = response
                predicted_output_all = self.lstm_model.predict(self.lstm_input, batch_size=1)
                predicted_output = predicted_output_all[0, 9, :2]

            # target location is at the maximum response. We must take into account the fact that, if
            # the target doesn't move, the peak will appear at the top-left corner, not at the centre
            # (this is discussed in the paper Fig. 6). The response map wrap around cyclically.
            v_centre, h_centre = np.unravel_index(self.response.argmax(), self.response.shape)
            self.vert_delta, self.horiz_delta = [v_centre - self.response.shape[0] / 2,
                                                 h_centre - self.response.shape[1] / 2]
            self.pos_old = [
                self.pos[1] + self.patch_size[1] * 1.0 / self.resize_size[1] * self.horiz_delta - self.target_sz[
                    1] / 2.,
                self.pos[0] + self.patch_size[0] * 1.0 / self.resize_size[0] * self.vert_delta - self.target_sz[
                    0] / 2., ]

            self.pos = [self.pos[0] + self.target_sz[0] * predicted_output[0],
                        self.pos[1] + self.target_sz[1] * predicted_output[1]]

            self.pos = [max(self.target_sz[0] / 2, min(self.pos[0], self.im_sz[0] - self.target_sz[0] / 2)),
                        max(self.target_sz[1] / 2, min(self.pos[1], self.im_sz[1] - self.target_sz[1] / 2))]
        elif self.feature_type == 'cnn':
            # We need to normalise it (because our training did so):
            response = self.response
            response = (response-response.min())/(response.max()-response.min())
            response = np.expand_dims(np.expand_dims(response, axis=0), axis=0)
            predicted_output = self.cnn_model.predict(response, batch_size=1)

            # target location is at the maximum response. We must take into account the fact that, if
            # the target doesn't move, the peak will appear at the top-left corner, not at the centre
            # (this is discussed in the paper Fig. 6). The response map wrap around cyclically.
            v_centre, h_centre = np.unravel_index(self.response.argmax(), self.response.shape)
            self.vert_delta, self.horiz_delta = [v_centre - self.response.shape[0]/2, h_centre - self.response.shape[1]/2]
            self.pos_old = [self.pos[1] + self.patch_size[1] * 1.0 / self.resize_size[1] * self.horiz_delta - self.target_sz[1] / 2.,
                            self.pos[0] + self.patch_size[0] * 1.0 / self.resize_size[0] * self.vert_delta - self.target_sz[0] / 2.,]

            self.pos = [self.pos[0] + self.target_sz[0] * predicted_output[0][0],
                    self.pos[1] + self.target_sz[1] * predicted_output[0][1]]

            self.pos = [max(self.target_sz[0] / 2, min(self.pos[0], self.im_sz[0] - self.target_sz[0] / 2)),
                        max(self.target_sz[1] / 2, min(self.pos[1], self.im_sz[1] - self.target_sz[1] / 2))]

        elif self.feature_type == 'multi_cnn':
            response_all = np.zeros(shape=(5, self.resize_size[0], self.resize_size[1]))
            for i in range(len(self.response)):
                response_all[i, :, :] = imresize(self.response[i], size=self.resize_size)

            response_all = response_all.astype('float32') / 255. - 0.5
            self.response_all = response_all
            response_all = np.expand_dims(response_all, axis=0)
            predicted_output = self.multi_cnn_model.predict(response_all, batch_size=1)

            # target location is at the maximum response. We must take into account the fact that, if
            # the target doesn't move, the peak will appear at the top-left corner, not at the centre
            # (this is discussed in the paper Fig. 6). The response map wrap around cyclically.
            self.vert_delta, self.horiz_delta = \
                [self.target_sz[0] * predicted_output[0][0], self.target_sz[1] * predicted_output[0][1]]
            self.pos = [self.pos[0] + self.target_sz[0] * predicted_output[0][0],
                        self.pos[1] + self.target_sz[1] * predicted_output[0][1]]

            self.pos = [max(self.target_sz[0] / 2, min(self.pos[0], self.im_sz[0] - self.target_sz[0] / 2)),
                        max(self.target_sz[1] / 2, min(self.pos[1], self.im_sz[1] - self.target_sz[1] / 2))]

                ##################################################################################
            # we need to train the tracker again for scaling, it's almost the replicate of train
            ##################################################################################
            # calculate the new target size
            # scale_change = predicted_output[0][2:]
            # self.target_sz = np.multiply(self.target_sz, scale_change.mean())
            # we also require the target size to be smaller than the image size deivided by paddings
            # self.target_sz = [min(self.im_sz[0], self.target_sz[0]), min(self.im_sz[1] , self.target_sz[1])]
            # self.patch_size = np.multiply(self.target_sz, (1 + self.padding))

        ##################################################################################
        # we need to train the tracker again here, it's almost the replicate of train
        ##################################################################################
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        x_new = self.get_features()
        xf_new = self.fft2(x_new)
        if self.feature_type == 'multi_cnn':
            for i in range(len(x_new)):
                k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new[i], x_new[i])
                kf = self.fft2(k)
                alphaf_new = np.divide(self.yf[i], kf + self.lambda_value)
                self.x[i] = (1 - self.adaptation_rate) * self.x[i] + self.adaptation_rate * x_new[i]
                self.xf[i] = (1 - self.adaptation_rate) * self.xf[i] + self.adaptation_rate * xf_new[i]
                self.alphaf[i] = (1 - self.adaptation_rate) * self.alphaf[i] + self.adaptation_rate * alphaf_new
        else:
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new, x_new)
            kf = self.fft2(k)
            alphaf_new = np.divide(self.yf, kf + self.lambda_value)
            self.x = (1 - self.adaptation_rate) * self.x + self.adaptation_rate * x_new
            self.xf = (1 - self.adaptation_rate) * self.xf + self.adaptation_rate * xf_new
            self.alphaf = (1 - self.adaptation_rate) * self.alphaf + self.adaptation_rate * alphaf_new

        # we also require the bounding box to be within the image boundary
        self.res.append([min(self.im_sz[1] - self.target_sz[1], max(0, self.pos[1] - self.target_sz[1] / 2.)),
                         min(self.im_sz[0] - self.target_sz[0], max(0, self.pos[0] - self.target_sz[0] / 2.)),
                         self.target_sz[1], self.target_sz[0]])

        return self.pos

    def get_scale_sample(self, im, scaleFactors):
        """
        Extract a sample fro the scale filter at the current location and scale
        :param im:
        :return:
        """
        import cv2
        from hog import hog
        resized_im_array = np.zeros((len(self.scaleFactors), int(np.floor(self.first_target_sz[0]/4) * np.floor(self.first_target_sz[1]/4) * 31)))
        for i, s in enumerate(scaleFactors):
            patch_sz = np.floor(self.first_target_sz * s)
            im_patch = self.get_subwindow(im, self.pos, patch_sz)  # extract image
            im_patch_resized = imresize(im_patch, self.first_target_sz)  #resize image to model size
            img_gray = cv2.cvtColor(im_patch_resized, cv2.COLOR_BGR2GRAY)
            features_hog, hog_image = hog(img_gray, orientations=31, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
            resized_im_array[i, :] = np.multiply(features_hog.flatten(), self.scale_window[i])

        return resized_im_array

    def dense_gauss_kernel(self, sigma, xf, x, zf=None, z=None, feature_correlation=None):
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
        if self.feature_type == 'raw' or self.feature_type == 'dsst':
            if len(xyf.shape) == 3:
                xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
            elif len(xyf.shape) == 2:
                xyf_ifft = np.fft.ifft2(xyf)
            # elif len(xyf.shape) == 4:
            #     xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=3))
        elif self.feature_type == 'hog':
            xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50' \
                or self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn' or self.feature_type =='multi_cnn':
            if feature_correlation is None:
                xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
            else:
                xyf_ifft = np.fft.ifft2(np.sum(np.multiply(xyf, self.feature_correlation[None, None, :]), axis=2))

        row_shift, col_shift = np.floor(np.array(xyf_ifft.shape) / 2).astype(int)
        xy_complex = np.roll(xyf_ifft, row_shift, axis=0)
        xy_complex = np.roll(xy_complex, col_shift, axis=1)
        c = np.real(xy_complex)
        d = np.real(xx) + np.real(zz) - 2 * c
        k = np.exp(-1. / sigma**2 * np.maximum(0, d) / N)

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

        # check for out-of-bounds coordinates and set them to the values at the borders
        ys[ys < 0] = 0
        ys[ys >= self.im_sz[0]] = self.im_sz[0] - 1

        xs[xs < 0] = 0
        xs[xs >= self.im_sz[1]] = self.im_sz[1] - 1

        # extract image

        if self.feature_type == 'raw' or self.feature_type == 'dsst':
            out = im[np.ix_(ys, xs)]
            # introduce scaling, here, we need them to be the same size
            if np.all(self.first_patch_sz == out.shape[:2]):
                return out
            else:
                out = imresize(out, self.first_patch_sz)
                return out / 255.
        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50' or \
             self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn' or self.feature_type == 'multi_cnn':
            c = np.array(range(3))
            out = im[np.ix_(c, ys, xs)]
            # if self.feature_type == 'vgg_rnn' or self.feature_type == 'cnn':
            #     from keras.applications.vgg19 import preprocess_input
            #     x = imresize(out.copy(), self.resize_size)
            #     out = np.multiply(x, self.cos_window_patch[:, :, None])
            return out

    def fft2(self, x):
        """
        FFT transform of the first 2 dimension
        :param x: M*N*C the first two dimensions are used for Fast Fourier Transform
        :return:  M*N*C the FFT2 of the first two dimension
        """
        if type(x) == list:
            x = [np.fft.fft2(f, axes=(0,1)) for f in x]
            return x
        else:
            return np.fft.fft2(x, axes=(0, 1))

    def get_features(self):
        """
        :param im: input image
        :return:
        """
        if self.feature_type == 'raw':
            #using only grayscale:
            if len(self.im_crop.shape) == 3:
                if self.sub_feature_type == 'gray':
                    img_gray = np.mean(self.im_crop, axis=2)
                    img_gray = img_gray - img_gray.mean()
                    features = np.multiply(img_gray, self.cos_window)
                else:
                    img_colour = self.im_crop - self.im_crop.mean()
                    features = np.multiply(img_colour, self.cos_window[:, :, None])
                return features

        elif self.feature_type == 'dsst':
            img_colour = self.im_crop - self.im_crop.mean()
            features = np.multiply(img_colour, self.cos_window[:, :, None])
            return features

        elif self.feature_type == 'vgg' or self.feature_type == 'resnet50':
            if self.feature_type == 'vgg':
                from keras.applications.vgg19 import preprocess_input
            elif self.feature_type == 'resnet50':
                from keras.applications.resnet50 import preprocess_input
            x = np.expand_dims(self.im_crop.copy(), axis=0)
            x = preprocess_input(x)
            features = self.extract_model.predict(x)
            features = np.squeeze(features)
            features = (features.transpose(1, 2, 0) - features.min()) / (features.max() - features.min())
            features = np.multiply(features, self.cos_window[:, :, None])
            return features

        elif self.feature_type == 'vgg_rnn' or self.feature_type=='cnn':
            from keras.applications.vgg19 import preprocess_input
            x = imresize(self.im_crop.copy(), self.resize_size)
            x = x.transpose((2, 0, 1)).astype(np.float64)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = self.extract_model.predict(x)
            features = np.squeeze(features)
            features = (features.transpose(1, 2, 0) - features.min()) / (features.max() - features.min())
            features = np.multiply(features, self.cos_window[:, :, None])
            return features

        elif self.feature_type == "multi_cnn":
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

    def train_rnn(self, frame, im, init_rect, target_sz, img_rgb_next, next_rect, next_target_sz):

        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        # Duh OBT is the reverse
        self.target_sz = target_sz[::-1]
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.im_sz = im.shape[1:]

        if frame==0:
            self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
            self.x = self.get_features()
            self.xf = self.fft2(self.x)
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x)
            self.alphaf = np.divide(self.yf, self.fft2(k) + self.lambda_value)

        ###################### Next frame #####################################
        self.im_crop = self.get_subwindow(img_rgb_next, self.pos, self.patch_size)
        z = self.get_features()
        zf = self.fft2(z)
        k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x, zf, z)
        kf = self.fft2(k)
        self.response = np.real(np.fft.ifft2(np.multiply(self.alphaf, kf)))
        ##################################################################################
        # we need to train the tracker again here, it's almost the replicate of train
        ##################################################################################
        self.pos_next = [next_rect[1] + next_rect[3] / 2., next_rect[0] + next_rect[2] / 2.]
        self.im_crop = self.get_subwindow(img_rgb_next, self.pos_next, self.patch_size)
        x_new = self.get_features()
        xf_new = self.fft2(x_new)
        k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new, x_new)
        kf = self.fft2(k)
        alphaf_new = np.divide(self.yf, kf + self.lambda_value)
        self.x = (1 - self.adaptation_rate) * self.x + self.adaptation_rate * x_new
        self.xf = (1 - self.adaptation_rate) * self.xf + self.adaptation_rate * xf_new
        self.alphaf = (1 - self.adaptation_rate) * self.alphaf + self.adaptation_rate * alphaf_new


        lstm_input = self.response.flatten()
        lstm_input.resize(1, np.prod(self.response_size))
        pos_move = np.array([(self.pos_next[0] - self.pos[0]), (self.pos_next[1] - self.pos[1])])
        pos_move.resize(1, 2)
        self.lstm_model.fit(lstm_input, pos_move, batch_size=1, verbose=1, nb_epoch=1, shuffle=False)
        print('Predicting')
        predicted_output = self.lstm_model.predict(lstm_input, batch_size=1)
        print(pos_move)
        print(predicted_output)

    def train_cnn(self, frame, im, init_rect, img_rgb_next, next_rect, x_train, y_train, count):

        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        # Duh OBT is the reverse
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.next_target_sz = np.asarray(next_rect[2:])
        self.next_target_sz = self.next_target_sz[::-1]
        self.scale_change = np.divide(np.array(self.next_target_sz).astype(float), self.target_sz)
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.im_sz = im.shape[1:]

        if frame == 0:
            self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
            self.x = self.get_features()
            self.xf = self.fft2(self.x)
            # if self.feature_type == 'multi_cnn':
            #     self.feature_correlation = []
            #     for i in range(len(self.x)):
            #         corr = np.multiply(self.x[i], self.y[i][:, :, None])
            #         corr = np.sum(np.sum(corr, axis=0), axis=0)
            #         # we compute the correlation of a filter within a layer to its features
            #         self.feature_correlation.append((corr - corr.min()) / (corr.max() - corr.min()))
            # here self.xf is list
            self.alphaf = []
            for i in range(len(self.x)):
                k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i])
                self.alphaf.append(np.divide(self.yf[i], self.fft2(k) + self.lambda_value))

        ###################### Next frame #####################################
        self.im_crop = self.get_subwindow(img_rgb_next, self.pos, self.patch_size)
        z = self.get_features()
        zf = self.fft2(z)
        self.response = []
        for i in range(len(z)):
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i], zf[i], z[i])
            kf = self.fft2(k)
            self.response.append(np.real(np.fft.ifft2(np.multiply(self.alphaf[i], kf))))

        ##################################################################################
        # we need to train the tracker again here, it's almost the replicate of train
        ##################################################################################
        self.pos_next = [next_rect[1] + next_rect[3] / 2., next_rect[0] + next_rect[2] / 2.]
        self.im_crop = self.get_subwindow(img_rgb_next, self.pos_next, self.patch_size)
        x_new = self.get_features()
        xf_new = self.fft2(x_new)
        for i in range(len(x_new)):
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new[i], x_new[i])
            kf = self.fft2(k)
            alphaf_new = np.divide(self.yf[i], kf + self.lambda_value)
            self.x[i] = (1 - self.adaptation_rate) * self.x[i] + self.adaptation_rate * x_new[i]
            self.xf[i] = (1 - self.adaptation_rate) * self.xf[i] + self.adaptation_rate * xf_new[i]
            self.alphaf[i] = (1 - self.adaptation_rate) * self.alphaf[i] + self.adaptation_rate * alphaf_new

        response_all = np.zeros(shape=(5, self.resize_size[0], self.resize_size[1]))
        for i in range(len(self.response)):
            response_all[i, :, :] = imresize(self.response[i], size=self.resize_size)

        x_train[count, :, :, :] = response_all
        self.pos_next = [next_rect[1] + next_rect[3] / 2., next_rect[0] + next_rect[2] / 2.]
        pos_move = np.array([(self.pos_next[0] - self.pos[0]) * 1.0 / self.target_sz[0],
                             (self.pos_next[1] - self.pos[1]) * 1.0 / self.target_sz[1]])
        y_train[count, :] = np.concatenate([pos_move, self.scale_change])
        count += 1
        return x_train, y_train, count

        # ('feature time:', 0.07054710388183594)
        # ('fft2:', 0.22904396057128906)
        # ('guassian kernel + fft2: ', 0.20537400245666504)

    def grabcut(self, im, init_rect):
        """
         :param im: image should be of 3 dimension: M*N*C
         :param pos: the centre position of the target
         :param target_sz: target size
         """
        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        self.res.append(init_rect)
        # Duh OBT is the reverse
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.im_sz = im.shape[1:]
        ########################################################
        # let's try grabcut now!
        ########################################################
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        import cv2
        from matplotlib.patches import Rectangle
        # sz = self.target_sz.astype('uint8')
        chosen_size = 2
        sz = (self.patch_size * (chosen_size / 3.2))
        im_crop = self.get_subwindow(im, self.pos, sz)
        img = im_crop.transpose(1, 2, 0).astype('uint8')

        coeff = 1.8
        rect = tuple(np.array([sz[::-1] / 2 - self.target_sz[::-1] / 2 * coeff, self.target_sz[::-1] * coeff]).astype(np.uint8).flatten())

        mask = np.zeros(img.shape[:2], dtype='uint8')
        bgdModel = np.zeros((1, 65))
        fgdModel = np.zeros((1, 65))

        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img_mask = img * mask2[:, :, np.newaxis]

        plt.figure(1)
        plt.clf()

        plt.subplot(221)
        plt.imshow(self.im_crop.transpose(1, 2, 0).astype('uint8'))
        plt.title('original image patch')

        tracking_figure_axes = plt.subplot(222)
        tracking_rect = Rectangle(
            xy=(rect[0], rect[1]),
            width=rect[2],
            height=rect[3],
            facecolor='none',
            edgecolor='r',
        )
        tracking_figure_axes.add_patch(tracking_rect)
        plt.imshow(img)
        plt.title('original image')

        plt.subplot(223)
        plt.imshow(img_mask)
        plt.title('grabcut image')

        plt.draw()
        plt.waitforbuttonpress(0.5)
