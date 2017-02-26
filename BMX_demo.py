"""
author: DI WU
stevenwudi@gmail.com
"""
from __future__ import print_function
from KCFpy_saliency import KCFTracker
import os
import numpy as np
import time
import json

img_dir = './data/bmx'
saliency_dir = './data/bmx_saliency_map'

with open('./data/bmx_sloth.json') as jf:
    data = json.load(jf)

gtRect = []
for i in range(len(data)):
    gtRect.append([int(data[i]['annotations'][0]['x']), int(data[i]['annotations'][0]['y']),
                   int(data[i]['annotations'][0]['width']), int(data[i]['annotations'][0]['height'])])


def main(debug=True):
    tracker = KCFTracker(feature_type='multi_cnn', sub_feature_type='dsst',
                           sub_sub_feature_type='adapted_lr', load_model=True, vgglayer='',
                           model_path='./trained_models/CNN_Model_OBT100_multi_cnn_best_cifar_big_valid.h5',
                           name_suffix='_best_valid_CNN', saliency_method=1, spatial_reg=1)

    from keras.preprocessing import image
    from visualisation_utils import plot_tracking_rect, show_precision
    img_list = sorted(os.listdir(img_dir))

    start_time = time.time()
    start_frame = 1
    tracker.res = []
    for frame in range(1, len(img_list)):
        image_path = os.path.join(img_dir, img_list[frame])
        img_rgb = image.load_img(image_path)
        img_rgb = image.img_to_array(img_rgb)

        img_saliency_path = os.path.join(saliency_dir, img_list[frame])
        img_saliency = image.load_img(img_saliency_path)
        img_saliency = image.img_to_array(img_saliency)
        if frame == start_frame:
            tracker.train(img_rgb, gtRect[start_frame], 'BMX', img_saliency)
        else:
            tracker.detect(img_rgb, frame, img_saliency)

        if debug and frame > start_frame:
            print("Frame ==", frame)
            print('horiz_delta: %.2f, vert_delta: %.2f' % (tracker.horiz_delta, tracker.vert_delta))
            print("pos", np.array(tracker.res[-1]).astype(int))
            print("gt", gtRect[frame])
            print("\n")
            plot_tracking_rect(frame + 0, img_rgb, tracker, gtRect[1:])

    total_time = time.time() - start_time
    tracker.fps = len(tracker.res) / total_time
    print("Frames-per-second:", tracker.fps)

    if True:
        tracker.precisions = show_precision(np.array(tracker.res), np.array(gtRect[1:]), 'BMX', wait_second=0)

    res = {'type': tracker.type, 'res': tracker.res, 'fps': tracker.fps}
    return tracker, res


if __name__ == "__main__":
    main()
