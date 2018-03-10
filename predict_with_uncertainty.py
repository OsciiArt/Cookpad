# just a example
# use it in each script

import numpy as np
import keras.backend as K
from keras import Model
from keras.layers import Dense, Input

def get_model(num_class):
    input = Input([5,])()
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
model = Model

f = K.function([model.layers[0].input, K.learning_phase()],
               [model.layers[-1].output])

def predict_with_uncertainty(f, x, n_iter=10):
    result = np.zeros((n_iter,) + x.shape)

    for iter in range(n_iter):
        result[iter] = f(x, 1)

    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)
    return prediction, uncertainty