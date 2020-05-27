from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, Activation, BatchNormalization, Add, AveragePooling2D, Input, ZeroPadding2D, concatenate, GlobalAveragePooling2D, Lambda


def conv2d(X, num_filters, filter_size, strides=1, padding="same", activation=True, name=None, kernel_initializer=None, bias_initializer=None):
    X = Conv2D(num_filters, filter_size, strides=strides, padding=padding,
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(X)
    X = BatchNormalization(axis=3, scale=False)(X)
    if activation:
        X = Activation("relu")(X)
    return X


def Stem(input_layer, kernel_init, bias_init, name=None):
    X = conv2d(input_layer, 32, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_init,
               bias_initializer=bias_init)

    X = conv2d(X, 32, (3, 3), strides=1, padding="valid", activation="relu", kernel_initializer=kernel_init,
               bias_initializer=bias_init)

    X = conv2d(X, 64, (3, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
               bias_initializer=bias_init)

    X1 = MaxPool2D((3, 3), strides=(2, 2), padding="valid")(X)

    X2 = conv2d(X, 96, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X = concatenate([X1, X2], axis=3)

    X1 = conv2d(X, 64, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X1 = conv2d(X1, 96, (3, 3), strides=1, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)
    X2 = conv2d(X, 64, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 64, (7, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 64, (1, 7), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 96, (3, 3), strides=1, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X = concatenate([X1, X2], axis=3)

    X1 = conv2d(X, 192, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = MaxPool2D((3, 3), strides=(2, 2), padding="valid")(X)

    X = concatenate([X1, X2], axis=3, name=name)

    return X


def InceptionResNet_A(X, kernel_init, bias_init, scale=1, name=None):
    X_shortcut = X

    X1 = conv2d(X, 32, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X, 32, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 32, (3, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X3 = conv2d(X, 32, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X3 = conv2d(X3, 48, (3, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X3 = conv2d(X3, 64, (3, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X_inception = concatenate([X1, X2, X3], axis=3)

    X_inception = conv2d(X_inception, 384, (1, 1), strides=1, padding="same", activation=False,
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init)

    final_layer = Lambda(lambda inputs: inputs[0] + inputs[1] * scale, name=name + "_Scaling_Residual")([X_shortcut, X_inception])

    final_layer = Activation("relu", name=name)(final_layer)

    return final_layer


def InceptionResNet_B(X, kernel_init, bias_init, scale=1, name=None):
    X_shortcut = X

    X1 = conv2d(X, 192, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X, 128, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 160, (1, 7), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 192, (7, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X_inception = concatenate([X1, X2], axis=3)

    X_inception = conv2d(X_inception, 1152, (1, 1), strides=1, padding="same", activation=False,
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init)

    final_layer = Lambda(lambda inputs: inputs[0] + inputs[1] * scale, name=name + "_Scaling_Residual")(
        [X_shortcut, X_inception])

    final_layer = Activation("relu", name=name)(final_layer)

    return final_layer


def InceptionResNet_C(X, kernel_init, bias_init, scale=1, name=None):
    X_shortcut = X

    X1 = conv2d(X, 192, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X, 192, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 224, (1, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 256, (3, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X_inception = concatenate([X1, X2], axis=3)

    X_inception = conv2d(X_inception, 2048, (1, 1), strides=1, padding="same", activation=False,
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init)

    final_layer = Lambda(lambda inputs: inputs[0] + inputs[1] * scale, name=name + "_Scaling_Residual")(
        [X_shortcut, X_inception])

    final_layer = Activation("relu", name=name)(final_layer)

    return final_layer


def Reduction_A(X, kernel_init, bias_init, name=None):
    X1 = MaxPool2D((3, 3), strides=(2, 2), padding="valid")(X)

    X2 = conv2d(X, 384, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X3 = conv2d(X, 256, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X3 = conv2d(X3, 256, (3, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X3 = conv2d(X3, 384, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X = concatenate([X1, X2, X3], axis=3, name=name)

    return X


def Reduction_B(X, kernel_init, bias_init, name=None):
    X1 = MaxPool2D((3, 3), strides=(2, 2), padding="valid")(X)

    X2 = conv2d(X, 256, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X2 = conv2d(X2, 384, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X3 = conv2d(X, 256, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X3 = conv2d(X3, 256, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X4 = conv2d(X, 256, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X4 = conv2d(X4, 256, (3, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X4 = conv2d(X4, 256, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_init,
                bias_initializer=bias_init)

    X = concatenate([X1, X2, X3, X4], axis=3, name=name)

    return X


if __name__ == '__main__':
    pass
