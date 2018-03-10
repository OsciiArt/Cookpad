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
                 early_stopping
                 ]
    return Callbacks



def swish(x):
    return x * K.sigmoid(x)


from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D

def get_model(num_class):
    base_model = DenseNet121(weights=None, include_top=False,
                       input_shape=[input_size,input_size,3], classes=1)
    print(base_model.summary())
    x = base_model.get_layer("bn").output
    # x = base_model.get_layer("block5_pool").output

    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_class, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
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
                  optimizer=optimizer,
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
    # img = randomHueSaturationValue(img,
    #                                hue_shift_limit=(-50, 50),
    #                                sat_shift_limit=(-5, 5),
    #                                val_shift_limit=(-15, 15),
    #                                u=0.)
    # img = randomShiftScaleRotate(img,
    #                              shift_limit=(-0.2, 0.2),
    #                              scale_limit=(-0.2, 0.5),
    #                              rotate_limit=(-30, 30),
    #                              aspect_limit=(-0.2, 0.2),
    #                              u=0.)
    # img = randomHorizontalFlip(img)
    # img = randomVerticalFlip(img)
    return img

def train_generator(x_train, y_train, img_dir, batch_size, shuffle=True):
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

        batch_x = pool.map(load_img,
                           [img_dir + '/{}'.format(x_train[id])
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


def get_mixer(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3):
    def mixer(img1, img2, mask1, mask2):
        img_h, img_w, img_c = img1.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return img1, mask1

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
        mask1[top:top + h, left:left + w, :] = mask2[top:top + h, left:left + w, :]

        return img1, mask1

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




def test_generator(x_train, img_dir, batch_size, shuffle, flip, flop, rot90=False):
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
            if flip:
                img = img[:,::-1]
            if flop:
                img = img[::-1]
            if rot90:
                img = np.transpose(img, [1,0,2])
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
    save_path = base_name + "_seed" + str(seed)
    model_path = "_"

    # Load data
    X_train, y_train, X_test = load_data()

    # CV
    ids_train_split, ids_valid_split = train_test_split(np.arange(X_train.shape[0]),
                                                        random_state=42, test_size=0.05,
                                                        stratify=y_train)


    # data process
    X_train_cv = X_train[ids_train_split]
    y_train_cv = y_train[ids_train_split]
    X_holdout = X_train[ids_valid_split]
    Y_holdout = y_train[ids_valid_split]


    number = np.zeros([600])
    for i in range(600):
        print(str(X_holdout[i]))
        print(os.path.splitext(str(X_holdout[i])))
        number[i] = int(os.path.splitext(str(X_holdout[i]))[0][6:])
    # print(X_train_cv.head())
    import shutil
    # for i in range(X_holdout.shape[0]):
    #     shutil.copyfile("input/train/" + X_holdout[i], "input/valid/" + X_holdout[i])

    # define file path and get callbacks
    weight_path = "model/" + save_path + '.hdf5'
    callbacks = get_callbacks(weight_path, patience=16)
    gen_val = train_generator(X_holdout, Y_holdout, "input/train", batch_size, shuffle=False)
    model = get_model(num_class)
    model2 = get_model2(model, num_class)
    model2.load_weights(filepath="model/train_180218_1_Dense_7th_training_ts0644.hdf5")

    f = K.function([model2.layers[0].input, K.learning_phase()],
                   [model2.layers[-1].output])

    def predict_with_uncertainty(f, x, n_iter=10):
        # print(x.shape)
        result = np.zeros((n_iter,x.shape[0], num_class))

        for iter in range(n_iter):
            result[iter] = f([x, 1])[0]

        prediction = result.mean(axis=0)
        uncertainty = result.var(axis=0)
        return prediction, uncertainty

    def prediction(f, batchgenerator):
        num_step = int(np.ceil(X_holdout.shape[0] / batch_size))
        pred_mean = np.zeros([X_holdout.shape[0], num_class], np.float32)
        pred_var = np.zeros([X_holdout.shape[0], num_class], np.float32)
        for i in range(num_step):
            batch = next(batchgenerator)
            mean, var = predict_with_uncertainty(f, batch)
            if i ==num_step-1:
                # print('here 2')
                pred_mean[i * batch_size:] = mean
                pred_var[i * batch_size:] = var
            else:
                # print('here 1')
                pred_mean[i * batch_size:(i+1)*batch_size] = mean
                pred_var[i * batch_size:(i+1)*batch_size] = var
        return pred_mean, pred_var

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

    # Getting Test prediction
    # Getting Test prediction
    gen_tst_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False,
                                   flip=False, flop=False)
    pred_mean1, pred_var1 = prediction(f, gen_tst_pred)
    gen_tst_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False,
                                   flip=True, flop=False)
    pred_mean2, pred_var2 = prediction(f, gen_tst_pred)
    gen_tst_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False,
                                   flip=True, flop=True)
    pred_mean3, pred_var3 = prediction(f, gen_tst_pred)
    gen_tst_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False,
                                   flip=False, flop=False)
    pred_mean4, pred_var4 = prediction(f, gen_tst_pred)

    gen_tst_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False,
                                   flip=False, flop=False, rot90=True)
    pred_mean5, pred_var5 = prediction(f, gen_tst_pred)
    gen_tst_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False,
                                   flip=True, flop=False, rot90=True)
    pred_mean6, pred_var6 = prediction(f, gen_tst_pred)
    gen_tst_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False,
                                   flip=True, flop=True, rot90=True)
    pred_mean7, pred_var7 = prediction(f, gen_tst_pred)
    gen_tst_pred = test_generator(X_holdout, "input/train", batch_size, shuffle=False,
                                   flip=False, flop=False, rot90=True)
    pred_mean8, pred_var8 = prediction(f, gen_tst_pred)

    pred_mean = (pred_mean1 + pred_mean2 + pred_mean3 + pred_mean4
                 + pred_mean5 + pred_mean6+ pred_mean7 + pred_mean8) / 8
    pred_var = (pred_var1 + pred_var2 + pred_var3 + pred_var4
                 + pred_var5 + pred_var6+ pred_var7 + pred_var8) / 8


    category_name = pd.read_csv("input/master.tsv", delimiter="\t", index_col=False, header=None).as_matrix()[:,0]

    pred_class_sorted = np.argsort(-pred_mean, axis=1)[:, :5]
    print(pred_class_sorted.shape)
    pred_proba_sorted = -np.sort(-pred_mean, axis=1)[:, :5]
    print(pred_proba_sorted.shape)
    pred_var_sorted = np.zeros([600,5])
    for i in range(600):
        pred_var_sorted[i] = pred_var[i][pred_class_sorted[i]]
    print(pred_var_sorted.shape)
    pred_name_sorted = category_name[pred_class_sorted]
    print(pred_name_sorted.shape)


    submission = pd.DataFrame({'id': X_holdout,
                               'number': number,
                               'true': category_name[Y_holdout],
                               'class1': pred_name_sorted[:,0],
                               'proba1': pred_proba_sorted[:,0],
                               'class2': pred_name_sorted[:,1],
                               'proba2': pred_proba_sorted[:,1],
                               'class3': pred_name_sorted[:,2],
                               'proba3': pred_proba_sorted[:,2],
                               'class4': pred_name_sorted[:,3],
                               'proba4': pred_proba_sorted[:,3],
                               'class5': pred_name_sorted[:,4],
                               'proba5': pred_proba_sorted[:,4],
                               'var1': pred_var_sorted[:,0],
                               'var2': pred_var_sorted[:,1],
                               'var3': pred_var_sorted[:,2],
                               'var4': pred_var_sorted[:,3],
                               'var5': pred_var_sorted[:,4],

                               })
    submit_path = "output/submission" + save_path + "valid_augx8_val_loss" + str(score[0]) + "_val_acc" + str(score[1]) + "_drop.csv"
    submission.to_csv(submit_path, index=False)

    # np.save("input/" + base_name + "_test.npy", pred_test)


def main():
    train(epochs=250, seed=0)


if __name__ == "__main__": main()
