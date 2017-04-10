import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches


if False:
    cnn_model = load_model('cnn_translation_scale.h5')
    # we first visualise thefigure to see whether it's meaningful to learn it
    arrow_length_unit = np.array(f["x_train"][0].shape)
    arrow_start_pos = np.array(f["x_train"][0].shape) / 2
    fig = plt.figure(1)
    plt.cla()

    max_arrow_width = 2
    max_head_width = max_arrow_width
    max_head_length = max_arrow_width

        # we also draw the scale changes
    for i in range(250,300):
        plt.cla()
        plt.imshow(f["x_train"][i, :, :])
        #print('y_train', np.multiply(f["y_train"][i][0:2], arrow_length_unit))
        print('y_train', f["y_train"][i])
        print('y_train_scale', np.multiply(f["y_train"][i][2:], arrow_length_unit))
        # now we plot arrow
        arrow_scale = np.multiply(f["y_train"][i][0:2], arrow_length_unit)
        #arrow_length = np.mean(np.sqrt(np.square(np.multiply(f["y_train"][i][0:2], arrow_length_unit))))
        arrow_length = 1 / 3.2

        width = max_arrow_width * arrow_length
        head_width = max_head_width
        head_length = max_head_length


        plt.arrow(arrow_start_pos[1], arrow_start_pos[0], arrow_scale[1]*arrow_length, arrow_scale[0] * arrow_length, fc='r', ec='w',
                  width=width, head_width=head_width, head_length=head_length)

        original_ellipse = Ellipse(xy=(arrow_start_pos[::-1]), width=20, height=20, facecolor='none', edgecolor='r')
        #fig.gca().add_artist(original_ellipse)

        next_ellipse = Ellipse(xy=(arrow_start_pos[::-1]), width=20*f["y_train"][i][3], height=20*f["y_train"][i][2], facecolor='none', edgecolor='g')
        #fig.gca().add_artist(next_ellipse)

        red_label = mpatches.Patch(color='red', label='original_ellipse')
        green_label = mpatches.Patch(color='green', label='next_ellipse')
        #plt.legend(handles=[red_label, green_label])

        ###########################################################
        # now let's visualise the prediction
        response = f["x_train"][i, :, :]
        response = np.expand_dims(np.expand_dims(response, axis=0), axis=0)
        predicted_output = cnn_model.predict(response, batch_size=1)

        title_label = mpatches.Patch(color='red', label='translation')
        plt.legend(handles=[title_label])
        plt.title("frame is %d" %i)
        plt.draw()
        plt.waitforbuttonpress(1)

''' List for the images
Starting benchmark for 1 trackers, evalTypes : ['OPE']
1_cnn, 1_BasketBall:1/1 - OPE
('Frames-per-second:', 144.58015344080573)
count 724
1_cnn, 2_Biker:1/1 - OPE
('Frames-per-second:', 145.99143844789097)
count 865
1_cnn, 3_Bird1:1/1 - OPE
('Frames-per-second:', 170.1960359083482)
count 1272
1_cnn, 4_BlurBody:1/1 - OPE
('Frames-per-second:', 121.39895496489025)
count 1605
1_cnn, 5_BlurCar2:1/1 - OPE
('Frames-per-second:', 122.26093055474169)
count 2189
1_cnn, 6_BlurFace:1/1 - OPE
('Frames-per-second:', 131.3800471570632)
count 2681
1_cnn, 7_BlurOwl:1/1 - OPE
('Frames-per-second:', 118.87913979818167)
count 3311
1_cnn, 8_Bolt:1/1 - OPE
('Frames-per-second:', 124.46295892892338)
count 3660
1_cnn, 9_Box:1/1 - OPE
('Frames-per-second:', 150.8893251145759)
count 4820
1_cnn, 10_Car1:1/1 - OPE
('Frames-per-second:', 658.4005189569164)
count 5839
1_cnn, 11_Car4:1/1 - OPE
('Frames-per-second:', 582.832186648077)
count 6497
1_cnn, 12_CarDark:1/1 - OPE
('Frames-per-second:', 416.8997546779181)
count 6889
1_cnn, 13_CarScale:1/1 - OPE
('Frames-per-second:', 226.68364487782358)
count 7140
1_cnn, 14_ClifBar:1/1 - OPE
('Frames-per-second:', 400.3968432261875)
count 7611
1_cnn, 15_Couple:1/1 - OPE
('Frames-per-second:', 422.39231213407976)
count 7750
1_cnn, 16_Crowds:1/1 - OPE
('Frames-per-second:', 153.6328249323953)
count 8096
1_cnn, 17_David:1/1 - OPE
('Frames-per-second:', 379.8265338307199)
count 8566
1_cnn, 18_Deer:1/1 - OPE
('Frames-per-second:', 157.9418587136617)
count 8636
1_cnn, 19_Diving:1/1 - OPE
('Frames-per-second:', 380.34707233357346)
count 8850
1_cnn, 20_DragonBaby:1/1 - OPE
('Frames-per-second:', 180.57061850241126)
count 8962
1_cnn, 21_Dudek:1/1 - OPE
('Frames-per-second:', 242.99461944092158)
count 10106
1_cnn, 22_Football:1/1 - OPE
('Frames-per-second:', 345.20383080599265)
count 10467
1_cnn, 23_Freeman4:1/1 - OPE
('Frames-per-second:', 620.9865632453598)
count 10749
1_cnn, 24_Girl:1/1 - OPE
('Frames-per-second:', 977.1131004430011)
count 11248
1_cnn, 25_Human3:1/1 - OPE
('Frames-per-second:', 146.099521139788)
count 12945
1_cnn, 26_Human4-2:1/1 - OPE
('Frames-per-second:', 147.71566423853503)
count 13611
1_cnn, 27_Human6:1/1 - OPE
('Frames-per-second:', 153.37295620267213)
count 14402
1_cnn, 28_Human9:1/1 - OPE
('Frames-per-second:', 450.27354539431957)
count 14706
1_cnn, 29_Ironman:1/1 - OPE
('Frames-per-second:', 195.1856951475752)
count 14871
1_cnn, 30_Jump:1/1 - OPE
('Frames-per-second:', 382.26536109186225)
count 14992
1_cnn, 31_Jumping:1/1 - OPE
('Frames-per-second:', 534.2464741409789)
count 15304
1_cnn, 32_Liquor:1/1 - OPE
('Frames-per-second:', 158.79197032432268)
count 17044
1_cnn, 33_Matrix:1/1 - OPE
('Frames-per-second:', 168.054083456368)
count 17143
1_cnn, 34_MotorRolling:1/1 - OPE
('Frames-per-second:', 195.17568883869487)
count 17306
1_cnn, 35_Panda:1/1 - OPE
('Frames-per-second:', 406.49396357715415)
count 18305
1_cnn, 36_RedTeam:1/1 - OPE
('Frames-per-second:', 427.6472958989162)
count 20222
1_cnn, 37_Shaking:1/1 - OPE
('Frames-per-second:', 198.07929312862254)
count 20586
1_cnn, 38_Singer2:1/1 - OPE
('Frames-per-second:', 196.07674617214352)
count 20951
1_cnn, 39_Skating1:1/1 - OPE
('Frames-per-second:', 178.5194039479933)
count 21350
1_cnn, 40_Skating2-1:1/1 - OPE
('Frames-per-second:', 205.3428333101338)
count 21822
1_cnn, 41_Skating2-2:1/1 - OPE
('Frames-per-second:', 204.73796371245678)
count 22294
1_cnn, 42_Skiing:1/1 - OPE
('Frames-per-second:', 199.14708645640383)
count 22374
1_cnn, 43_Soccer:1/1 - OPE
('Frames-per-second:', 175.84326701609302)
count 22765
1_cnn, 44_Surfer:1/1 - OPE
('Frames-per-second:', 274.45040175649484)
count 23140
1_cnn, 45_Sylvester:1/1 - OPE
('Frames-per-second:', 695.8278894045917)
count 24484
1_cnn, 46_Tiger2:1/1 - OPE
('Frames-per-second:', 148.6716150945748)
count 24848
1_cnn, 47_Trellis:1/1 - OPE
('Frames-per-second:', 351.48754763424745)
count 25416
1_cnn, 48_Walking:1/1 - OPE
('Frames-per-second:', 112.45885401183146)
count 25827
1_cnn, 49_Walking2:1/1 - OPE
('Frames-per-second:', 353.8542567769194)
count 26326
1_cnn, 50_Woman:1/1 - OPE
('Frames-per-second:', 371.75716000604376)
count 26922

26922/26922 [==============================] - 22s - loss: 0.0027


'''