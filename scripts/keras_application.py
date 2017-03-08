from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.models import Model
import numpy as np

from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Embedding
from keras.layers import Convolution2D, MaxPooling2D, TimeDistributed
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributedDense

preds = np.zeros(shape=(1, 1000))
print('Predicted:', decode_predictions(preds, top=3)[0])


print('Creating Model')

model = Sequential()
model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'), input_shape=(10, 1, 60, 40) , name='convolution2d_1'))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Convolution2D(32, 3, 3),  name='convolution2d_2'))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))
model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same'), name='convolution2d_3'))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Convolution2D(64, 3, 3), name='convolution2d_4'))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))
model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(512), name='dense_1'))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Dropout(0.5)))

model.add(LSTM(512, return_sequences=True))
model.add(TimeDistributed(Dense(3)))

model.load_weights('cnn_translation_scale_combine.h5', by_name=True)
cnn_model = load_model('cnn_translation_scale_combine.h5')

for l in model.layers:
    print l.name

##### a dummy test to see whether our network can be trained properly!
X_train = np.random.rand(10, 10, 1, 60, 40)
y_train = np.random.rand(10, 10, 3)

model.compile(loss='mse', optimizer='adam')

print('Train...')
model.fit(X_train, y_train, batch_size=10, nb_epoch=1)

score = model.evaluate(X_train, y_train, batch_size=10)
print('Test score:', score)


##############################
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import LSTM, TimeDistributed, Embedding
            ##################################

from keras.models import Sequential
from keras.layers import Dense, LSTM

##################################
tsteps = 1
batch_size = 1
lstm_embedding_size = 64
lstm_target_size = 2

print('Creating Model')
lstm_model = Sequential()
lstm_model.add(Embedding(40*60, 128, batch_input_shape=(batch_size, 40*60)))
lstm_model.add(LSTM(lstm_embedding_size,
                         batch_input_shape=(batch_size, tsteps, 128),
                         return_sequences=False,
                         stateful=True))
lstm_model.add(Dense(lstm_target_size))
lstm_model.compile(loss='mse', optimizer='rmsprop')

# ##############################
# model = Sequential()
# model.add(LSTM(50,
#                batch_input_shape=(batch_size, tsteps, 1),
#                return_sequences=True,
#                stateful=True))
# model.add(LSTM(50,
#                batch_input_shape=(batch_size, tsteps, 1),
#                return_sequences=False,
#                stateful=True))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='rmsprop')


from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

img_path = '/home/stevenwudi/Documents/Python_Project/OBT/tracker_benchmark/data/Car1/img/0001.jpg'

# img = image.load_img(img_path, target_size=(150, 150))
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
base_model = ResNet50(weights='imagenet', include_top=True)
extract_model = Model(input=base_model.input, output=base_model.get_layer('merge_2').output)
preds = extract_model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

for layer in base_model.layers:
    print(layer.name)