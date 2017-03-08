from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np


model = VGG19(weights='imagenet')


x, y = result['res'][frame - 1][0], result['res'][frame - 1][1]
width = result['res'][frame - 1][2]
height = result['res'][frame - 1][3]

img_crop = img_rgb[:, y-int(height*0.5) :y + int(1.5*height), x-int(width*0.5):x+ int(1.5*width)]

from scipy.misc import imresize

im_sz = imresize(img_crop.transpose(1,2,0), (224, 224))
plt.imshow(im_sz)


x = np.expand_dims(im_sz.transpose(2,0,1), axis=0).astype('float64')
x = preprocess_input(x)
preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=10)[0])