import numpy as np # linear algebra
np.random.seed(42)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import time
import os, glob
import cv2

# parameters
format = "%H%M"
ts = time.strftime(format)
base_name = os.path.splitext(__file__)[0] + "_ts" + ts
input_size = 75


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GaussianNoise
from keras.layers import GlobalMaxPooling2D, Reshape, UpSampling3D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras import backend as K


def get_callbacks(save_path, lr=0.001, patience=64):
    csv_logger = CSVLogger(save_path + '_log.csv', append=True)
    # check_path = save_path + '_e{epoch:02d}_vl{val_loss:.5f}.hdf5'
    check_path = save_path
    save_checkpoint = ModelCheckpoint(filepath=check_path, monitor='val_loss', save_best_only=True)
    lerning_rate_schedular = ReduceLROnPlateau(patience=8, min_lr=lr * 0.00001)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=16,
                                   verbose=1,
                                   min_delta=1e-4,
                                   mode='min')
    Callbacks = [csv_logger,
                 save_checkpoint,
                 # lerning_rate_schedular,
                 # early_stopping
                 ]
    return Callbacks



def swish(x):
    return x * K.sigmoid(x)


from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD

def get_model(num_class):
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=[75,75,3], classes=1)
    x = base_model.get_layer('block5_pool').output

    x = GlobalMaxPooling2D()(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu', name='fc3')(x)
    x = Dropout(0.3)(x)

    predictions = Dense(num_class, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image) # sikisou, saido, meido
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image




def randomShiftScaleRotate(image,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))

    return image

def randomHorizontalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)

    return image

def randomHorizontalFlip_pseudo(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)

    return image


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def train_generator(x_train, y_train, img_dir, batch_size, shuffle=True):
    # x_train = x_train.as_matrix()
    # y_train = y_train.as_matrix()
    y_train = np.eye(55)[y_train]
    batch_index = 0
    n = x_train.shape[0]
    # print("n", n)
    eraser = get_random_eraser(v_h=0.)
    while 1:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        batch_x = []
        batch_id = index_array[current_index: current_index + current_batch_size]
        # print(batch_x_base)
        for id in batch_id:
            # print(x_train[0])
            # print(x_train[id])
            # print(img_dir + '/{}'.format(x_train[id]))

            img = cv2.imread(img_dir + '/{}'.format(x_train[id]))
            # print("img shape", img.shape)
            img = cv2.resize(img, (input_size, input_size))
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-5, 5),
                                           sat_shift_limit=(-1, 1),
                                           val_shift_limit=(-2, 2),
                                           u=0.5)
            img = randomShiftScaleRotate(img,
                                                shift_limit=(-0.2, 0.2),
                                                scale_limit=(-0.2, 0.2),
                                                rotate_limit=(-30, 30),
                                         aspect_limit = (-0.2, 0.2),
                                         u=0.5)
            img = randomHorizontalFlip(img)
            # img =eraser(img)
            batch_x.append(img)
        batch_x = np.array(batch_x, np.float32) / 255

        batch_y = y_train[index_array[current_index: current_index + current_batch_size]]

        # print("batch shape", batch_x.shape, batch_y.shape)

        yield (batch_x, batch_y)


def test_generator(x_train, img_dir, batch_size, shuffle=True):
    # x_train = x_train.as_matrix()
    # y_train = y_train.as_matrix()
    batch_index = 0
    n = x_train.shape[0]
    # print("n", n)
    eraser = get_random_eraser(v_h=0.)
    while 1:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        batch_x = []
        batch_id = index_array[current_index: current_index + current_batch_size]
        # print(batch_x_base)
        for id in batch_id:
            # print(x_train[0])
            # print(x_train[id])
            # print(img_dir + '/{}'.format(x_train[id]))

            img = cv2.imread(img_dir + '/{}'.format(x_train[id]))
            # print("img shape", img.shape)
            img = cv2.resize(img, (input_size, input_size))
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-5, 5),
                                           sat_shift_limit=(-1, 1),
                                           val_shift_limit=(-2, 2),
                                           u=0.5)
            img = randomShiftScaleRotate(img,
                                                shift_limit=(-0.2, 0.2),
                                                scale_limit=(-0.2, 0.2),
                                                rotate_limit=(-30, 30),
                                         aspect_limit = (-0.2, 0.2),
                                         u=0.5)
            img = randomHorizontalFlip(img)
            # img =eraser(img)
            batch_x.append(img)
        batch_x = np.array(batch_x, np.float32) / 255

        # batch_y = y_train[index_array[current_index: current_index + current_batch_size]]

        # print("batch shape", batch_x.shape, batch_y.shape)

        yield batch_x



def load_data(train_path="input/train_master.tsv", test_path="input/sample_submit.tsv"):
    train = pd.read_csv(train_path, delimiter="\t", index_col=False)
    test = pd.read_csv(test_path, delimiter="\t", index_col=False)
    print("train shape", train.shape)
    print(train.head())
    X_train = train['file_name'].as_matrix()
    y_train = train['category_id'].as_matrix()
    # y_train = np.eye(55)[y_train]
    # print(y_train[:5])
    # print(y_train.shape)
    X_test = test.iloc[:,0]

    return X_train, y_train, X_test


from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss


def train(epochs, seed):
    # parameter
    batch_size = 128
    num_class = 55
    save_path = base_name + "_seed" + str(seed)
    model_path = "_"

    # Load data
    X_train, y_train, X_test = load_data()

    # CV
    K = 2
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=seed).split(X_train, y_train))
    y_train_pred_log = np.empty([X_train.shape[0], num_class], dtype=np.float32)
    y_test_pred_log =  np.empty([K, X_test.shape[0], num_class], dtype=np.float32)

    for j, (train_idx, test_idx) in enumerate(folds):
        print('\n===================FOLD=', j)

        # data process
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout = y_train[test_idx]
        # print(X_train_cv.head())


        # define file path and get callbacks
        weight_path = "model/" + save_path + '.hdf5'
        callbacks = get_callbacks(weight_path, patience=128)
        gen = train_generator(X_train_cv, y_train_cv, "input/train", batch_size)
        gen_val = train_generator(X_holdout, Y_holdout, "input/train", batch_size, shuffle=False)
        gen_val_pred = test_generator(X_holdout, "input/train", batch_size)
        gen_tst_pred = test_generator(X_test, "input/test", batch_size)
        model = get_model(num_class)
        # model.fit_generator(generator=gen,
        #                     steps_per_epoch=np.ceil(X_train_cv.shape[0] / batch_size),
        #                     epochs=epochs,
        #                     verbose=1,
        #                     callbacks=callbacks,
        #                     validation_data=gen_val,
        #                     validation_steps=np.ceil(X_holdout.shape[0] / batch_size),
        #                     )
        #
        # # Getting the Best Model
        # model.load_weights(filepath=weight_path)

        # Getting Training Score
        # score = model.evaluate_generator(generator=gen_trn_eval,
        #                                  steps=np.ceil(X_train.shape[0]/batch_size))
        # print('Train loss:', score[0])
        # print('Train accuracy:', score[1])

        # Getting Valid Score
        score = model.evaluate_generator(generator=gen_val,
                                         steps=np.ceil(X_holdout.shape[0]/batch_size))
        print('Valid loss:', score[0])
        print('Valid accuracy:', score[1])

        # Getting validation prediction
        pred_valid = model.predict_generator(generator=gen_val_pred,
                                             steps=np.ceil(X_holdout.shape[0]/batch_size))
        y_train_pred_log[test_idx] = pred_valid

        # Getting Test prediction
        pred_test = model.predict_generator(generator=gen_tst_pred,
                                            steps=np.ceil(X_test.shape[0]/batch_size))
        y_test_pred_log[j] = pred_test

        submission = pd.DataFrame({'id': X_test, 'predict': np.argmax(y_test_pred_log[j], axis=1)})
        submit_path = "output/submission" + save_path+ "_fold" + str(j)  + "_val_loss" + str(score[0]) + "_val_acc" + str(score[1]) + ".csv"
        submission.to_csv(submit_path, index=False, header=False)

        # save weight
        # new_weight_path = "model/" + save_path + "_fold" + str(j) + "_val_loss" + str(score[0]) + "_val_acc" + str(score[1])+ ".hdf5"
        # os.rename(weight_path, new_weight_path)

    y_test_pred_log = np.mean(y_test_pred_log, axis=0)

    # save prediction
    cv_loss = log_loss(y_train, y_train_pred_log)
    print(np.argmax(y_train_pred_log, axis=1).shape)
    submission = pd.DataFrame({'id': X_train, 'predict': np.argmax(y_train_pred_log, axis=1)})
    submit_path = "output/pred_stage1_"+ save_path  + "_val_loss" + str(cv_loss) + ".csv"
    submission.to_csv(submit_path, index=False, header=False)

    print(np.argmax(y_test_pred_log, axis=1).shape)
    submission = pd.DataFrame({'id': X_test, 'predict': np.argmax(y_test_pred_log, axis=1)})
    submit_path = "output/submission_"+ save_path  + "_avr_val_loss" + str(cv_loss) + ".csv"
    submission.to_csv(submit_path, index=False, header=False)

    return model_path, weight_path, submit_path, cv_loss


def main():
    fold = 2
    submit_path_list = []
    cv = []
    for i in range(fold):
        _, _, submit_path, valid_score =train(epochs=1, seed=i)
        submit_path_list.append(submit_path)
        cv.append(valid_score)

    # ensamble
    prediction_list = []
    for i in range(fold):
        submit_path = submit_path_list[i]
        prediction = pd.read_csv(submit_path)['is_iceberg'].as_matrix()
        prediction = prediction.reshape([-1,1])
        prediction_list.append(prediction)

    predictions = np.concatenate(prediction_list, axis=1)
    predictions = np.mean(predictions, axis=1)
    test_id = pd.read_json("input/test.json")["id"]
    submission = pd.DataFrame({'id': test_id, 'is_iceberg': predictions})
    cv_mean = np.array(cv).mean()
    file_name = os.path.basename(os.path.splitext(__file__)[0])
    submission.to_csv("output/submission_"+ file_name + "_cv_score_" + str(cv_mean) + "_ensamble.csv", index=False)


if __name__ == "__main__": main()

