"""
This is a python reimplementation of the open source tracker in
High-Speed Tracking with Kernelized Correlation Filters
Joao F. Henriques, Rui Caseiro, Pedro Martins, and Jorge Batista, tPAMI 2015
modified by Di Wu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage.transform import resize


class KCFTracker:
    def __init__(self, feature_type='raw', sub_feature_type='', sub_sub_feature_type='',
                 debug=False, gt_type='rect', load_model=False, vgglayer='',
                 model_path='./trained_models/CNN_Model_OBT100_multi_cnn_final.h5',
                 cnn_maximum=False, name_suffix="", saliency_method=None, spatial_reg=0):
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
        self.sub_sub_feature_type = sub_sub_feature_type
        self.name = 'KCF' + feature_type
        self.fps = -1
        self.type = gt_type
        self.res = []
        self.im_sz = []
        self.debug = debug  # a flag indicating to plot the intermediate figures
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
                if vgglayer[:6]=='block2':
                    self.cell_size = 42
                elif vgglayer[:6]=='block3':
                    self.cell_size = 4
                elif vgglayer[:6] == 'block4':
                    self.cell_size = 8
                elif vgglayer[:6] == 'block5':
                    self.cell_size = 16
                else:
                    assert("not implemented")

                self.base_model = VGG19(include_top=False, weights='imagenet')
                self.extract_model = Model(input=self.base_model.input, output=self.base_model.get_layer('block3_conv4').output)
            elif self.feature_type == 'resnet50':
                from keras.applications.resnet50 import ResNet50
                from keras.models import Model
                self.base_model = ResNet50(weights='imagenet', include_top=False)
                self.extract_model = Model(input=self.base_model.input,
                                           output=self.base_model.get_layer('activation_10').output)

            self.feature_bandwidth_sigma = 1
            self.adaptation_rate = 0.01
            if self.sub_feature_type == 'grabcut':
                self.grabcut_mask_path = './figures/grabcut_masks/'
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
            self.adaptation_rate = 0.0025
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
                self.yf.append(self.fft2(y))

            # self.path_resize_size = np.multiply(self.yf.shape, (1 + self.padding))
            # self.cos_window_patch = np.outer(np.hanning(self.resize_size[0]), np.hanning(self.resize_size[1]))
            # Embedding
            if load_model:
                from keras.models import load_model
                if self.sub_feature_type=='class':
                    self.multi_cnn_model = load_model('./models/CNN_Model_OBT100_multi_cnn_best_valid_cnn_cifar_small_batchnormalisation_class_scale.h5')
                    from models.DataLoader import DataLoader
                    loader = DataLoader(batch_size=32, filename="./data/OBT100_new_multi_cnn%d.hdf5")
                    self.translation_value = np.asarray(loader.translation_value)
                    self.scale_value = np.asarray(loader.scale_value)
                else:
                    self.multi_cnn_model = load_model(model_path)
                    self.cnn_maximum = cnn_maximum
            if self.sub_feature_type=='dsst':
                # this method adopts from the paper  Martin Danelljan, Gustav Hger, Fahad Shahbaz Khan and Michael Felsberg.
                # "Accurate Scale Estimation for Robust Visual Tracking". (BMVC), 2014.
                # The project website is: http: // www.cvl.isy.liu.se / research / objrec / visualtracking / index.html
                self.scale_step = 1.01
                self.nScales = 33
                self.scaleFactors = self.scale_step ** (np.ceil(self.nScales * 1.0 / 2) - range(1, self.nScales + 1))
                self.scale_window = np.hanning(self.nScales)
                self.scale_sigma_factor = 1. / 4
                self.scale_sigma = self.nScales / np.sqrt(self.nScales) * self.scale_sigma_factor
                self.ys = np.exp(
                    -0.5 * ((range(1, self.nScales + 1) - np.ceil(self.nScales * 1.0 / 2)) ** 2) / self.scale_sigma ** 2)
                self.ysf = np.fft.fft(self.ys)
                self.min_scale_factor = []
                self.max_scale_factor = []
                self.xs = []
                self.xsf = []
                self.sf_num = []
                self.sf_den = []
                # we use linear kernel as in the BMVC2014 paper
                self.new_sf_num = []
                self.new_sf_den = []
                self.scale_response = []
                self.lambda_scale = 1e-2
                self.adaptation_rate_scale = 0.005

                if sub_sub_feature_type == 'adapted_lr':
                    self.sub_sub_feature_type = sub_sub_feature_type
                    self.acc_time = 5
                    self.loss = np.zeros(shape=(self.acc_time, 5))
                    self.loss_mean = np.zeros(shape=(self.acc_time, 5))
                    self.loss_std = np.zeros(shape=(self.acc_time, 5))
                    self.adaptation_rate_range = [0.005, 0.0]
                    self.adaptation_rate_scale_range = [0.005, 0.00]
                    self.adaptation_rate = self.adaptation_rate_range[0]
                    self.adaptation_rate_scale = self.adaptation_rate_scale_range[0]
                    self.stability = 1

        if self.sub_feature_type:
            self.name += '_'+sub_feature_type
            self.feature_correlation = None
            if self.sub_sub_feature_type:
                self.name += '_' + sub_sub_feature_type
            if self.cnn_maximum:
                self.name += '_cnn_maximum'


        self.saliency_method = saliency_method
        self.feature_correlation = None
        self.spatial_reg = spatial_reg

        self.name += name_suffix

    def train(self, im, init_rect, seqname, img_saliency_origin=None):
        """
        :param im: image should be of 3 dimension: M*N*C
        :param pos: the centre position of the target
        :param target_sz: target size
        """
        self.pos = [init_rect[1]+init_rect[3]/2., init_rect[0]+init_rect[2]/2.]
        self.res.append(init_rect)
        # for scaling, we always need to set it to 1
        self.currentScaleFactor = 1
        # Duh OBT is the reverse
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.first_target_sz = self.target_sz  # because we might introduce the scale changes in the detection
        # desired padded input, proportional to input target size
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.first_patch_sz = np.array(self.patch_size).astype(int)   # because we might introduce the scale changes in the detection
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

        if self.sub_feature_type == 'grabcut':
            import matplotlib.image as mpimg

            img_grabcut = mpimg.imread(self.grabcut_mask_path+seqname+".png")
            grabcut_shape = self.x.shape[:2]
            img_grabcut = resize(img_grabcut, grabcut_shape)
            corr = np.multiply(self.x, img_grabcut[:,:,None])
            corr = np.sum(np.sum(corr, axis=0), axis=0)
            # we compute the correlation of a filter within a layer to its features
            self.feature_correlation = (corr - corr.min()) / (corr.max() - corr.min())

        ############### saliency incorporation ##########################
        if self.feature_type == 'multi_cnn':
            # multi_cnn will render the models to be of a list

            if img_saliency_origin is not None:
                from skimage.transform import resize
                self.feature_correlation = []
                img_saliency_crop = self.get_subwindow(img_saliency_origin, self.pos, self.patch_size)
                self.img_saliency_crop = img_saliency_crop.mean(axis=0) / 255.
                for i in range(len(self.x)):
                    saliency_shape = self.x[i].shape[:2]
                    img_saliency = resize(self.img_saliency_crop, saliency_shape)
                    corr = np.multiply(self.x[i], img_saliency[:, :, None])
                    corr = np.sum(np.sum(corr, axis=0), axis=0)
                    # we compute the correlation of a filter within a layer to its features
                    self.feature_correlation.append((corr - corr.min()) / (corr.max() - corr.min()))
                    if self.saliency_method == 1:
                        self.x[i] *= self.feature_correlation[-1]
                        self.xf[i] *= self.feature_correlation[-1]

            self.alphaf = []
            for i in range(len(self.x)):
                k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[i], self.x[i])
                if self.spatial_reg:
                    salience_resize_f = self.fft2(resize((1-self.img_saliency_crop), k.shape))
                    reg = np.multiply(salience_resize_f, np.conj(salience_resize_f))
                    self.alphaf.append(np.divide(self.yf[i], self.fft2(k) + reg * 1./(np.prod(reg.shape))))
                    #self.alphaf.append(np.divide(self.yf[i], self.fft2(k) + reg * self.lambda_value))
                else:
                    self.alphaf.append(np.divide(self.yf[i], self.fft2(k) + self.lambda_value))

            if self.sub_feature_type == 'dsst':
                self.min_scale_factor = self.scale_step ** (np.ceil(np.log(max(5. / self.patch_size)) / np.log(self.scale_step)))
                self.max_scale_factor = self.scale_step ** (np.log(min(np.array(self.im_sz[:2]).astype(float) / self.target_sz)) / np.log(self.scale_step))
                self.xs = self.get_scale_sample(im, self.currentScaleFactor * self.scaleFactors)
                self.xsf = np.fft.fftn(self.xs, axes=[0])
                # we use linear kernel as in the BMVC2014 paper
                self.sf_num = np.multiply(self.ysf[:, None], np.conj(self.xsf))
                self.sf_den = np.real(np.sum(np.multiply(self.xsf, np.conj(self.xsf)), axis=1))


            ################ we plot the filters here:
            if False:
                from mpl_toolkits.mplot3d import Axes3D
                plt.figure()
                #filter = np.real(np.fft.ifft2(np.conj(self.alphaf[0])))
                filter = np.real(np.fft.ifft2(self.alphaf[0]))
                row_shift, col_shift = np.floor(np.array(filter.shape) / 2).astype(int)
                filter = np.roll(np.roll(filter, row_shift, axis=0), col_shift, axis=1)
                plt.imshow(filter)
                plt.colorbar()

                fig = plt.figure()
                ax = Axes3D(fig)
                X = np.arange(0, filter.shape[0])
                Y = np.arange(0, filter.shape[1])
                X, Y = np.meshgrid(X, Y)
                ax.plot_surface(Y, X, filter.transpose(1,0), rstride=2, cstride=2, cmap='rainbow')
                plt.show()

                plt.figure()
                plt.imshow(self.im_crop.transpose(1,2,0)/255.)

                plt.figure()
                salience_resize = resize(self.img_saliency_crop, filter.shape)
                plt.imshow(salience_resize)

                # plot the saliency regularisor map:
                fig2 = plt.figure()
                ax2 = Axes3D(fig2)
                X = np.arange(0, salience_resize.shape[0])
                Y = np.arange(0, salience_resize.shape[1])
                X, Y = np.meshgrid(X, Y)
                ax2.plot_surface(X, Y, (1-salience_resize.transpose(1,0)), rstride=2, cstride=2, cmap='rainbow')
                plt.show()

                plt.figure()
                k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf[0], self.x[0])
                plt.imshow(k)

        else:
            k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x)
            self.alphaf = np.divide(self.yf, self.fft2(k) + self.lambda_value)

    def detect(self, im, frame, img_saliency=None):
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

        if img_saliency is not None and self.saliency_method == 2 and self.feature_type == 'multi_cnn':
            self.feature_correlation = []
            from skimage.transform import resize
            img_saliency_crop = self.get_subwindow(img_saliency, self.pos, self.patch_size)
            self.img_saliency_crop = img_saliency_crop.mean(axis=0) / 255.
            for i in range(len(self.x)):
                saliency_shape = z[i].shape[:2]
                img_saliency = resize(self.img_saliency_crop, saliency_shape)
                corr = np.multiply(z[i], img_saliency[:, :, None])
                corr = np.sum(np.sum(corr, axis=0), axis=0)
                # we compute the correlation of a filter within a layer to its features
                self.feature_correlation.append((corr - corr.min()) / (corr.max() - corr.min()))
                z[i] *= self.feature_correlation[-1]
                zf[i] *= self.feature_correlation[-1]

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
            self.max_list = [np.max(x) for x in self.response]
            if self.sub_sub_feature_type == 'adapted_lr':
                loss_idx = np.mod(frame, self.acc_time)
                self.loss[loss_idx] = 1 - np.asarray(self.max_list)
                self.loss_mean = np.mean(self.loss, axis=0)
                self.loss_std = np.std(self.loss, axis=0)

                if frame > self.acc_time:
                    stability_coeff = np.abs(self.loss[loss_idx]-self.loss_mean) / self.loss_std
                    self.stability = np.mean(np.exp(-stability_coeff))
                    # stability value is small(0), object is stable, adaptive learning rate is increased to maximum
                    # stability value is big(1), object is not stable, adaptive learning rate is decreased to minimum
                    self.adaptation_rate = max(0, self.adaptation_rate_range[1] + \
                                           self.stability*(self.adaptation_rate_range[0] - self.adaptation_rate_range[1]))
                    self.adaptation_rate_scale = max(0, self.adaptation_rate_scale_range[1] + \
                                                 self.stability*(self.adaptation_rate_scale_range[0] - self.adaptation_rate_scale_range[1]))

            for i in range(len(self.response)):
                response_all[i, :, :] = imresize(self.response[i], size=self.resize_size)
                if self.sub_feature_type == 'class' or self.cnn_maximum:
                    response_all[i, :, :] = np.multiply(response_all[i, :, :], self.max_list[i])

            response_all = response_all.astype('float32') / 255. - 0.5
            self.response_all = response_all
            response_all = np.expand_dims(response_all, axis=0)
            predicted_output = self.multi_cnn_model.predict(response_all, batch_size=1)

            if self.sub_feature_type=='class':
                translational_x = np.dot(predicted_output[0], self.translation_value)
                translational_y = np.dot(predicted_output[1], self.translation_value)
                scale_change = np.dot(predicted_output[2], self.scale_value)
                # translational_x = self.translation_value[np.argmax(predicted_output[0])]
                # translational_y = self.translation_value[np.argmax(predicted_output[1])]
                # scale_change = self.scale_value[np.argmax(predicted_output[2])]
                # calculate the new target size
                self.target_sz = np.divide(self.target_sz, scale_change)
                # we also require the target size to be smaller than the image size deivided by paddings
                self.target_sz = [min(self.im_sz[0], self.target_sz[0]), min(self.im_sz[1], self.target_sz[1])]
                self.patch_size = np.multiply(self.target_sz, (1 + self.padding))

                self.vert_delta, self.horiz_delta = \
                    [self.target_sz[0] * translational_x, self.target_sz[1] * translational_y]

                self.pos = [self.pos[0] + self.target_sz[0] * translational_x,
                            self.pos[1] + self.target_sz[1] * translational_y]
                self.pos = [max(self.target_sz[0] / 2, min(self.pos[0], self.im_sz[0] - self.target_sz[0] / 2)),
                            max(self.target_sz[1] / 2, min(self.pos[1], self.im_sz[1] - self.target_sz[1] / 2))]
            else:
                ##################################################################################
                # we need to train the tracker again for scaling, it's almost the replicate of train
                ##################################################################################
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

        ##################################################################################
        # we need to train the tracker again here, it's almost the replicate of train
        ##################################################################################
        if self.sub_feature_type == 'dsst':
            self.xs = self.get_scale_sample(im, self.currentScaleFactor * self.scaleFactors)
            self.xsf = np.fft.fftn(self.xs, axes=[0])
            # calculate the correlation response of the scale filter
            scale_response_fft = np.divide(np.multiply(self.sf_num, self.xsf),
                                           (self.sf_den[:, None] + self.lambda_scale))
            scale_reponse = np.real(np.fft.ifftn(np.sum(scale_response_fft, axis=1)))
            recovered_scale = np.argmax(scale_reponse)
            # update the scale
            self.currentScaleFactor *= self.scaleFactors[recovered_scale]
            if self.currentScaleFactor < self.min_scale_factor:
                self.currentScaleFactor = self.min_scale_factor
            elif self.currentScaleFactor > self.max_scale_factor:
                self.currentScaleFactor = self.max_scale_factor
            # we only update the target size here.
            new_target_sz = np.multiply(self.currentScaleFactor, self.first_target_sz)
            self.pos -= (new_target_sz-self.target_sz)/2
            self.target_sz = new_target_sz
            self.patch_size = np.multiply(self.target_sz, (1 + self.padding))

        # we update the model from here
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        x_new = self.get_features()
        xf_new = self.fft2(x_new)

        if img_saliency is not None and self.saliency_method == 2 and self.feature_type == 'multi_cnn':
            for i in range(len(x_new)):
                x_new[i] *= self.feature_correlation[i]
                xf_new[i] *= self.feature_correlation[i]

        if self.feature_type == 'multi_cnn':
            for i in range(len(x_new)):
                k = self.dense_gauss_kernel(self.feature_bandwidth_sigma, xf_new[i], x_new[i])
                kf = self.fft2(k)

                if self.spatial_reg and img_saliency is not None and self.saliency_method == 2 and self.spatial_reg:
                    from skimage.transform import resize
                    salience_resize_f = self.fft2(resize((1-self.img_saliency_crop), k.shape))
                    reg = np.multiply(salience_resize_f, np.conj(salience_resize_f))
                    alphaf_new = np.divide(self.yf[i], kf + reg * 1./(np.prod(reg.shape)))
                else:
                    alphaf_new = np.divide(self.yf[i], kf + self.lambda_value)
                self.x[i] = (1 - self.adaptation_rate) * self.x[i] + self.adaptation_rate * x_new[i]
                self.xf[i] = (1 - self.adaptation_rate) * self.xf[i] + self.adaptation_rate * xf_new[i]
                self.alphaf[i] = (1 - self.adaptation_rate) * self.alphaf[i] + self.adaptation_rate * alphaf_new
            if self.sub_feature_type == 'dsst':
                self.xs = self.get_scale_sample(im, self.currentScaleFactor * self.scaleFactors)
                self.xsf = np.fft.fftn(self.xs, axes=[0])
                # we use linear kernel as in the BMVC2014 paper
                new_sf_num = np.multiply(self.ysf[:, None], np.conj(self.xsf))
                new_sf_den = np.real(np.sum(np.multiply(self.xsf, np.conj(self.xsf)), axis=1))
                self.sf_num = (1 - self.adaptation_rate_scale) * self.sf_num + self.adaptation_rate * new_sf_num
                self.sf_den = (1 - self.adaptation_rate_scale) * self.sf_den + self.adaptation_rate * new_sf_den

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
            xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))

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

        elif self.feature_type == 'dsst':
            img_colour = self.im_crop - self.im_crop.mean()
            features = np.multiply(img_colour, self.cos_window[:, :, None])

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
                if self.saliency_method==1 and self.feature_correlation is not None:
                    features = np.multiply(features, self.feature_correlation[i][None, None, :])
                features_list[i] = np.multiply(features, self.cos_window[i][:, :, None])
            return features_list
        else:
            assert 'Non implemented!'

        return features

    def get_scale_sample(self, im, scaleFactors):
        from pyhog import pyhog
        resized_im_array = []
        for i, s in enumerate(scaleFactors):
            patch_sz = np.floor(self.first_target_sz * s)
            im_patch = self.get_subwindow(im, self.pos, patch_sz)  # extract image
            im_patch_resized = imresize(im_patch, self.first_target_sz)  #resize image to model size
            features_hog = pyhog.features_pedro(im_patch_resized.astype(np.float64)/255.0, 4)
            resized_im_array.append(np.multiply(features_hog.flatten(), self.scale_window[i]))

        return np.asarray(resized_im_array)

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

    def grabcut(self, im, init_rect, seq_name):
        """
         :param im: image should be of 3 dimension: M*N*C
         :param pos: the centre position of the target
         :param target_sz: target size
         """
        import cv2
        global img, img2, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over
        BLUE = [255, 0, 0]  # rectangle color
        RED = [0, 0, 255]  # PR BG
        GREEN = [0, 255, 0]  # PR FG
        BLACK = [0, 0, 0]  # sure BG
        WHITE = [255, 255, 255]  # sure FG

        DRAW_BG = {'color': BLACK, 'val': 0}
        DRAW_FG = {'color': WHITE, 'val': 1}
        DRAW_PR_FG = {'color': GREEN, 'val': 3}
        DRAW_PR_BG = {'color': RED, 'val': 2}

        # setting up flags
        rect = (0, 0, 1, 1)
        drawing = False  # flag for drawing curves
        rectangle = False  # flag for drawing rect
        rect_over = False  # flag to check if rect drawn
        rect_or_mask = 0  # flag for selecting rect or mask mode
        value = DRAW_FG  # drawing initialized to FG
        thickness = 3  # brush thickness

        def onmouse(event, x, y, flags, param):
            global img, img2, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over

            # Draw Rectangle
            # if event == cv2.EVENT_RBUTTONDOWN:
            #     rectangle = True
            #     ix, iy = x, y
            #
            # elif event == cv2.EVENT_MOUSEMOVE:
            #     if rectangle == True:
            #         img = img2.copy()
            #         cv2.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            #         rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
            #         rect_or_mask = 0
            #
            # elif event == cv2.EVENT_RBUTTONUP:
            #     rectangle = False
            #     rect_over = True
            #     cv2.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            #     rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
            #     rect_or_mask = 0
            #     print(" Now press the key 'n' a few times until no further change \n")

            # draw touchup curves
            if event == cv2.EVENT_LBUTTONDOWN:
                rect_over = True
                if rect_over == False:
                    print("first draw rectangle \n")
                else:
                    drawing = True
                    cv2.circle(img, (x, y), thickness, value['color'], -1)
                    cv2.circle(mask, (x, y), thickness, value['val'], -1)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing == True:
                    cv2.circle(img, (x, y), thickness, value['color'], -1)
                    cv2.circle(mask, (x, y), thickness, value['val'], -1)

            elif event == cv2.EVENT_LBUTTONUP:
                if drawing == True:
                    drawing = False
                    cv2.circle(img, (x, y), thickness, value['color'], -1)
                    cv2.circle(mask, (x, y), thickness, value['val'], -1)

        self.pos = [init_rect[1] + init_rect[3] / 2., init_rect[0] + init_rect[2] / 2.]
        self.res.append(init_rect)
        # Duh OBT is the reverse
        self.target_sz = np.asarray(init_rect[2:])
        self.target_sz = self.target_sz[::-1]
        self.patch_size = np.floor(self.target_sz * (1 + self.padding))
        self.first_patch_sz = np.array(self.patch_size).astype(int)
        self.im_sz = im.shape[:2]
        ########################################################
        # let's try grabcut now!
        ########################################################
        self.im_crop = self.get_subwindow(im, self.pos, self.patch_size)
        sz = np.array(self.im_crop.shape[:2])
        img = self.get_subwindow(im, self.pos, sz)
        img2 = img.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
        output = np.zeros(img.shape, np.uint8)  # output image to be shown

        #####################################################
        coeff = 1.5
        rect = np.array([sz[::-1] / 2 - self.target_sz[::-1] / 2 * coeff, sz[::-1] / 2 + self.target_sz[::-1] / 2 * coeff]).astype(np.int).flatten()

        # input and output windows
        cv2.namedWindow('output')
        cv2.namedWindow('input')
        cv2.setMouseCallback('input', onmouse)
        cv2.moveWindow('input', img.shape[1] + 10, 90)
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), BLUE, 2)

        while True:
            cv2.imshow('output', output)
            cv2.imshow('input', img)
            k = 0xFF & cv2.waitKey(1)
            # key bindings
            if k == 27:  # esc to exit
                break
            elif k == ord('0'):  # BG drawing
                print(" mark background regions with left mouse button \n")
                value = DRAW_BG
            elif k == ord('1'):  # FG drawing
                print(" mark foreground regions with left mouse button \n")
                value = DRAW_FG
            elif k == ord('2'):  # PR_BG drawing
                value = DRAW_PR_BG
            elif k == ord('3'):  # PR_FG drawing
                value = DRAW_PR_FG
            elif k == ord('s'):  # save image
                bar = np.zeros((img.shape[0], 5, 3), np.uint8)
                res = np.hstack((img2, bar, img, bar, output))
                #cv2.imwrite('./figures/grabcut_output.png', res)
                cv2.imwrite('./figures/masks/'+seq_name+'.png', mask2)
                print(" Result saved as image \n")
                cv2.destroyAllWindows()
                break
            elif k == ord('r'):  # reset everything
                print("resetting \n")
                rect = (0, 0, 1, 1)
                drawing = False
                rectangle = False
                rect_or_mask = 100
                rect_over = False
                value = DRAW_FG
                img = img2.copy()
                mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
                output = np.zeros(img.shape, np.uint8)  # output image to be shown
            elif k == ord('n'):  # segment the image
                print(""" For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n""")
                if (rect_or_mask == 0):  # grabcut with rect
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    rect_tuple = (rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1])
                    cv2.grabCut(img2, mask, rect_tuple, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
                    rect_or_mask = 1
                elif rect_or_mask == 1:  # grabcut with mask
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    cv2.grabCut(img2, mask, rect_tuple, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

            mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
            output = cv2.bitwise_and(img2, img2, mask=mask2)

