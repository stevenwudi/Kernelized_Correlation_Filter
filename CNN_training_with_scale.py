import numpy as np
from keras.optimizers import SGD
from models.CNN_CIFAR import cnn_cifar_small_batchnormalisation_class_scale
from models.DataLoader import DataLoader
from keras.models import load_model
from scripts.progress_bar import printProgress
from time import time, localtime
# this is a predefined dataloader
loader = DataLoader(batch_size=32, filename="./data/OBT100_new_multi_cnn%d.hdf5")

# construct the model here (pre-defined model)
# model = cnn_cifar_small_batchnormalisation_class_scale(loader.image_shape,
#                                                        len(loader.translation_value),
#                                                        len(loader.scale_value))

model = load_model('./models/CNN_Model_OBT100_multi_cnn_best_valid_cnn_cifar_small_batchnormalisation_class_scale.h5')
model.summary()

nb_epoch = 200
early_stopping = True
early_stopping_count = 0
early_stopping_wait = 3
train_loss = []
translation_x_loss = []
translation_y_loss = []
scale_loss = []
valid_loss = []
valid_translation_x_loss = []
valid_translation_y_loss = []
valid_scale_loss = []
learning_rate = [1e-06 * 10**x for x in range(3)]

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=learning_rate[-1], decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              loss_weights=[1., 1., 1.])

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
#               loss_weights=[1., 1., 1.])

# load validation data from the h5py file (heavy lifting here)
x_valid, translation_x_class_valid, translation_y_class_valid,  y_scale_class_valid =\
    loader.get_valid_class()
best_valid = np.inf

for e in range(nb_epoch):
    print("epoch %d" % e)
    loss_list = []
    time_list = []
    time_start = time()
    for i in range(loader.n_iter_train):
        time_start_batch = time()
        X_batch, translation_x_class, translation_y_class, y_scale_class = \
            loader.next_train_batch_with_scale_buckets()
        l1, l2, l3, l4 = model.train_on_batch({'main_input': X_batch},
                                              {'translation_x_class': translation_x_class,
                                               'translation_y_class': translation_y_class,
                                               'y_scale_class': y_scale_class})
        loss_list.append(l1)
        translation_x_loss.append(l2)
        translation_y_loss.append(l3)
        scale_loss.append(l4)

        # calculate some time information
        time_list.append(time() - time_start_batch)
        eta = (loader.n_iter_train - i) * np.array(time_list).mean()
        printProgress(i, loader.n_iter_train-1, prefix='Progress:',
                      suffix='batch error: %0.5f, translation_x_loss: %0.5f, translation_y_loss: %0.5f, scale_loss: %0.5f. ETA: %0.2f sec.'
                             %(np.asarray(loss_list).mean(), np.asarray(translation_x_loss).mean(),
                               np.asarray(translation_y_loss).mean(), np.asarray(scale_loss).mean(), eta), barLength=50)
    train_loss.append(np.asarray(loss_list).mean())
    print('training loss is %f, one epoch uses: %0.2f sec' % (train_loss[-1], time() - time_start))
    l1, l2, l3, l4 = model.evaluate({'main_input': x_valid},
                                     {'translation_x_class': translation_x_class_valid,
                                      'translation_y_class': translation_y_class_valid,
                                      'y_scale_class': y_scale_class_valid})
    valid_loss.append(l1)
    valid_translation_x_loss.append(l2)
    valid_translation_y_loss.append(l3)
    valid_scale_loss.append(l4)
    print('Valid error: %0.5f, translation_x_loss: %0.5f, translation_y_loss: %0.5f, scale_loss; %0.5f.'
          %(np.array(valid_loss).mean(), np.asarray(valid_translation_x_loss).mean(),
            np.asarray(valid_translation_y_loss).mean(), np.asarray(valid_scale_loss).mean()))

    if best_valid > valid_loss[-1]:
        early_stopping_count = 0
        print('saving best valid result...')
        best_valid = valid_loss[-1]
        model.save('./trained_models/CNN_Model_OBT100_multi_cnn_best_valid_'+model.name+'.h5')
    else:
        # we wait for early stopping loop until a certain time
        early_stopping_count += 1
        if early_stopping_count > early_stopping_wait:
            early_stopping_count = 0
            if len(learning_rate) > 1:
                learning_rate.pop()
                print('decreasing the learning rate to: %f'%learning_rate[-1])
                model.optimizer.lr.set_value(learning_rate[-1])
            else:
                break

lt = localtime()
lt_str = str(lt.tm_year)+"."+str(lt.tm_mon).zfill(2)+"." \
            +str(lt.tm_mday).zfill(2)+"."+str(lt.tm_hour).zfill(2)+"."\
            +str(lt.tm_min).zfill(2)+"."+str(lt.tm_sec).zfill(2)
np.savetxt('./trained_models/train_loss_'+model.name+'_'+lt_str+'.txt', train_loss)
np.savetxt('./trained_models/valid_loss_'+model.name+'_'+lt_str+'.txt', valid_loss)
model.save('./trained_models/CNN_Model_OBT100_multi_cnn_'+model.name+'_final.h5')
print("done")

#### we show some visualisation here
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
train_loss = np.loadtxt('./trained_models/train_loss_'+model.name+'_'+lt_str+'.txt')
valid_loss = np.loadtxt('./trained_models/valid_loss_'+model.name+'_'+lt_str+'.txt')

plt.plot(train_loss, 'b')
plt.plot(valid_loss, 'r')

blue_label = mpatches.Patch(color='blue', label='train_loss')
red_label = mpatches.Patch(color='red', label='valid_loss')
plt.legend(handles=[blue_label, red_label])

