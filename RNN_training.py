
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Convolution2D, MaxPooling2D, TimeDistributed
from keras.layers import Dense, Dropout, Activation, Flatten
import h5py
import numpy as np

print('Creating Model')
freez_conv = True
if not freez_conv:
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'), input_shape=(10, 1, 60, 40), name='convolution2d_1'))
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
    model.add(TimeDistributed(Dense(4)))

else:
    print('convolutional layers are freezed')
    model = Sequential()
    model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'), input_shape=(10, 1, 60, 40),
                              name='convolution2d_1'), trainable=False)
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Convolution2D(32, 3, 3), name='convolution2d_2'), trainable=False)
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, border_mode='same'), name='convolution2d_3'), trainable=False)
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Convolution2D(64, 3, 3), name='convolution2d_4'), trainable=False)
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512), name='dense_1'), trainable=False)
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dense(2)))


model.load_weights('cnn_translation_scale_combine.h5', by_name=True)


###
print('loading data')
f_load = h5py.File("OBT50_scale.hdf5", "r")
X_train = f_load["x_train"][:, :, :, :]
X_train = np.expand_dims(X_train, axis=2)
y_train = f_load["y_train"][:, :, :]
print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)

for i in range(X_train.shape[0]):
    X_train[i, :, :, :] = (X_train[i, :, :, :] - X_train[i, :, :, :].min()) / (
    X_train[i, :, :, :].max() - X_train[i, :, :, :].min())
###
print('start training...')

optimizer = RMSprop(lr=0.001, decay=0.95)
model.compile(loss='mse', optimizer=optimizer)

model.fit(X_train, y_train, batch_size=16, nb_epoch=10)

print "done"
model.save('rnn_translation_scale.h5')

################
# testing
###########

# from keras.models import load_model
# lstm_model = load_model('rnn_translation_scale.h5')
# input_temp = X_train[0, 0, 0, :, :]
# input_tile = np.broadcast_to(input_temp, (10, 60, 40))
# input_tile = np.expand_dims(np.expand_dims(input_tile, axis=1), axis=0)
# prediction = lstm_model.predict(input_tile, batch_size=1)
#
# input_temp = X_train[0, :, 0, :, :]
# input_tile = np.expand_dims(np.expand_dims(input_temp, axis=1), axis=0)
# prediction = lstm_model.predict(input_tile, batch_size=1)
# y_train[0,:]input_temp = X_train[0, :, 0, :, :]
# input_tile = np.expand_dims(np.expand_dims(input_temp, axis=1), axis=0)
# prediction = lstm_model.predict(input_tile, batch_size=1)
# y_train[0,:]