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
input_size = 221


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GaussianNoise
from keras.layers import GlobalMaxPooling2D, Reshape, UpSampling3D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K


def get_callbacks(save_path, lr=0.001, patience=64):
    csv_logger = CSVLogger(save_path + '_log.csv', append=True)
    # check_path = save_path + '_e{epoch:02d}_vl{val_loss:.5f}.hdf5'
    check_path = save_path
    save_checkpoint = ModelCheckpoint(filepath=check_path, monitor='val_loss', save_best_only=True)
    lerning_rate_schedular = ReduceLROnPlateau(patience=8, min_lr=lr * 0.00001)
    def lrs(epoch):
        if epoch<100:
            return 1e-4
        elif epoch<200:
            return 1e-5
        else:
            return 1e-6

    learning_rate_Schedular = LearningRateScheduler(lambda epoch: lrs(epoch))
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=16,
                                   verbose=1,
                                   min_delta=1e-4,
                                   mode='min')
    Callbacks = [csv_logger,
                 save_checkpoint,
                 learning_rate_Schedular,
                 # early_stopping
                 ]
    return Callbacks



def swish(x):
    return x * K.sigmoid(x)


from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.optimizers import SGD, Adam
from keras.layers import GlobalAveragePooling2D

def get_model(num_class):
    base_model = DenseNet121(weights=None, include_top=False,
                       input_shape=[input_size,input_size,3], classes=1)
    # print(base_model.summary())
    x = base_model.get_layer("bn").output
    # x = base_model.get_layer("block5_pool").output

    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_class, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

from keras.layers import Multiply

def get_model2(base_model, num_class):
    x = base_model.get_layer("bn").output # 7,7,1024
    feature = Conv2D(128, (3,3), activation='relu', name='feature')(x) # TODO num_channel
    gate = Conv2D(128, (3,3), activation='sigmoid', name='gate')(x)
    x = Multiply() ([feature, gate])
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_class, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.0001)
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

def randomVerticalFlip(image, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)

    return image


def get_mixer(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3):
    def mixer(img1, img2, y1, y2):
        img_h, img_w, img_c = img1.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return img1, y1

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        img1[top:top + h, left:left + w, :] = img2[top:top + h, left:left + w, :]


        return img1, mask1

    return mixer



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


from multiprocessing import Pool

def load_img(args):
    img_path = args
    img = cv2.imread(img_path)
    # print("img shape", img.shape)
    img = cv2.resize(img, (input_size, input_size))
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-50, 50),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15),
                                   u=0.25)
    img = randomShiftScaleRotate(img,
                                 shift_limit=(-0.2, 0.2),
                                 scale_limit=(-0.2, 0.5),
                                 rotate_limit=(-30, 30),
                                 aspect_limit=(-0.2, 0.2),
                                 u=0.25)
    img = randomHorizontalFlip(img)
    img = randomVerticalFlip(img)
    return img


def load_img_valid(args):
    img_path = args
    img = cv2.imread(img_path)
    img = cv2.resize(img, (input_size, input_size))
    return img

def train_generator(x_train, y_train, img_dir, batch_size, shuffle=True):
    # x_train = x_train.as_matrix()
    # y_train = y_train.as_matrix()
    y_train = np.eye(55)[y_train]
    batch_index = 0
    n = x_train.shape[0]
    # print("n", n)
    eraser = get_random_eraser(v_h=255)
    pool = Pool(16)
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

        batch_id = index_array[current_index: current_index + current_batch_size]

        batch_x = pool.map(load_img,
                           ['{}'.format(x_train[id])
                           for id in batch_id])
        for id in range(len(batch_x)):
            img = batch_x[id]
            # img =eraser(img)
            # img =eraser(img)
            # img =eraser(img)
            # img =eraser(img)
            # img =eraser(img)
            batch_x[id] = img
        batch_x = np.array(batch_x, np.float32) / 255

        batch_y = y_train[index_array[current_index: current_index + current_batch_size]]

        # print("batch shape", batch_x.shape, batch_y.shape)

        yield (batch_x, batch_y)


def valid_generator(x_train, y_train, img_dir, batch_size, shuffle=True):
    # x_train = x_train.as_matrix()
    # y_train = y_train.as_matrix()
    y_train = np.eye(55)[y_train]
    batch_index = 0
    n = x_train.shape[0]
    # print("n", n)
    eraser = get_random_eraser(v_h=0.)
    pool = Pool(4)
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

        batch_id = index_array[current_index: current_index + current_batch_size]

        batch_x = pool.map(load_img_valid,
                           ['{}'.format(x_train[id])
                           for id in batch_id])
        for id in range(len(batch_x)):
            img = batch_x[id]
            img =eraser(img)
            # img =eraser(img)
            # img =eraser(img)
            # img =eraser(img)
            # img =eraser(img)
            batch_x[id] = img
        batch_x = np.array(batch_x, np.float32) / 255

        batch_y = y_train[index_array[current_index: current_index + current_batch_size]]

        # print("batch shape", batch_x.shape, batch_y.shape)

        yield (batch_x, batch_y)


def get_mixer(p=1, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3):
    def mixer(img1, img2, y1, y2):
        img_h, img_w, img_c = img1.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return img1, y1

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        img1[top:top + h, left:left + w, :] = img2[top:top + h, left:left + w, :]
        y = (1- h/img_h*w/img_w) * y1 + h/img_h*w/img_w * y2

        return img1, y

    return mixer

def mix_generator(X_train, Y_train, img_dir, batch_size, shuffle=True):
    alpha = 0.2
    gen1 = train_generator(X_train, Y_train, img_dir, batch_size, shuffle)
    gen2 = train_generator(X_train, Y_train, img_dir, batch_size, shuffle)
    while True:
        batch1 = next(gen1)
        batch2 = next(gen2)
        current_batch_size = batch1[0].shape[0]
        l = np.random.beta(alpha, alpha, current_batch_size)
        X_l = l.reshape(current_batch_size, 1, 1, 1)
        Y_l = l.reshape(current_batch_size, 1)

        batch_x = batch1[0] * X_l + batch2[0] * (1 - X_l)
        batch_y = batch1[1] * Y_l + batch2[1] * (1 - Y_l)

        yield (batch_x, batch_y)



def mix_generator2(X_train, Y_train, img_dir, batch_size, shuffle=True):
    gen1 = mix_generator(X_train, Y_train, img_dir, batch_size, shuffle)
    gen2 = mix_generator(X_train, Y_train, img_dir, batch_size, shuffle)
    mixer =get_mixer()
    while True:
        batch1 = next(gen1)
        batch2 = next(gen2)
        batch_x = []
        batch_y = []
        for i in range(batch1[0].shape[0]):
            x1, y1 = batch1[0][i], batch1[1][i]
            x2, y2 = batch2[0][i], batch2[1][i]
            new_x, new_y = mixer(x1, x2, y1, y2)
            batch_x.append(new_x)
            batch_y.append(new_y)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch = (batch_x, batch_y)

        yield batch



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

            img = cv2.imread('{}'.format(x_train[id]))
            # print("img shape", img.shape)
            img = cv2.resize(img, (input_size, input_size))
            # img =eraser(img)
            batch_x.append(img)
        batch_x = np.array(batch_x, np.float32) / 255

        # batch_y = y_train[index_array[current_index: current_index + current_batch_size]]

        # print("batch shape", batch_x.shape, batch_y.shape)

        yield batch_x



def load_data(train_path="input/train_master.tsv", test_path="input/sample_submit.tsv"):
    train = pd.read_csv(train_path, delimiter="\t", index_col=False)
    test = pd.read_csv(test_path, delimiter="\t", index_col=False, header=None)
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
    batch_size = 64
    num_class = 55
    save_path = base_name
    model_path = "_"

    # Load data
    X_train, y_train, X_test = load_data()
    x_test_base = pd.read_csv("output/pred_test_180210_1_Dense_5th_training.csv",
                         index_col=False)
    print(x_test_base.shape)
    x_test_base = x_test_base[x_test_base.proba1 < -0.95]
    X_test = x_test_base['id'].as_matrix()
    y_test = x_test_base['label'].as_matrix().astype(np.uint8)
    print(X_test.shape)
    X_train = "input/train/" + X_train
    X_test = "input/test/" + X_test
    print(X_train[:10])
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # CV
    ids_train_split, ids_valid_split = train_test_split(np.arange(X_train.shape[0]),
                                                        random_state=42, test_size=0.05,
                                                        stratify=y_train)


    # data process
    X_train_cv = X_train[ids_train_split]
    y_train_cv = y_train[ids_train_split]
    X_train_cv = np.concatenate([X_train_cv, X_test])
    y_train_cv = np.concatenate([y_train_cv, y_test])
    X_holdout = X_train[ids_valid_split]
    Y_holdout = y_train[ids_valid_split]
    # print(X_train_cv.head())


    # define file path and get callbacks
    weight_path = "model/" + save_path + '.hdf5'
    callbacks = get_callbacks(weight_path, patience=16)
    gen = train_generator(X_train_cv, y_train_cv, "input/train", batch_size)
    gen_val = valid_generator(X_holdout, Y_holdout, "input/train", batch_size, shuffle=False)
    gen_val_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False)
    gen_tst_pred = test_generator(X_test, "input/test", batch_size, shuffle=False)
    model = get_model(num_class)
    # model.load_weights(filepath="model/train_180217_1_Dense_6th_training_ts0304.hdf5")
    print(model.summary())
    model2 = get_model2(model, num_class)
    model2.load_weights(filepath="model/train_180218_1_Dense_7th_training_ts0644.hdf5")

    model2.fit_generator(generator=gen,
                        steps_per_epoch=np.ceil(X_train_cv.shape[0] / batch_size),
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=gen_val,
                        validation_steps=np.ceil(X_holdout.shape[0] / batch_size),
                        )

    # Getting the Best Model
    model2.save_weights(filepath=weight_path[:-4] + "_nostop.hdf5")
    model2.load_weights(filepath=weight_path)

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

    # Getting Test prediction
    pred_test = model.predict_generator(generator=gen_tst_pred,
                                        steps=np.ceil(X_test.shape[0]/batch_size))

    submission = pd.DataFrame({'id': X_test, 'predict': np.argmax(pred_test, axis=1)})
    submit_path = "output/submission" + save_path + "_val_loss" + str(score[0]) + "_val_acc" + str(score[1]) + ".tsv"
    submission.to_csv(submit_path, index=False, header=False, sep='\t')

    np.save("input/" + base_name + "_valid.npy", pred_valid)
    np.save("input/" + base_name + "_test.npy", pred_test)


def main():
    train(epochs=300, seed=0)


if __name__ == "__main__": main()

