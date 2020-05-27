from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, Activation, BatchNormalization, Add, AveragePooling2D, Input, ZeroPadding2D, concatenate, GlobalAveragePooling2D, Lambda


def conv2d_bn(X, num_filters, filter_size, strides=1, padding="same", activation=None, name=None, kernel_initializer=None, bias_initializer=None):
    X = Conv2D(num_filters, filter_size, strides=strides, padding=padding,
               kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(X)
    X = BatchNormalization(axis=3, scale=False)(X)
    if activation:
        try:
            X = Activation(activation)(X)
        except Exception as e:
            raise ValueError("Wrong activation parameter!")
        return X


def Stem(input_layer, kernel_initializer=None, bias_initializer=None, name=None):
    X = conv2d_bn(input_layer, 32, (3, 3), strides=2, padding="valid", activation="relu",
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer)

    X = conv2d_bn(X, 32, (3, 3), strides=1, padding="valid", activation="relu", kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer)

    X = conv2d_bn(X, 64, (3, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer)

    X1 = MaxPool2D((3, 3), strides=(2, 2), padding="valid")(X)

    X2 = conv2d_bn(X, 96, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X = concatenate([X1, X2], axis=3)

    X1 = conv2d_bn(X, 64, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X1 = conv2d_bn(X1, 96, (3, 3), strides=1, padding="valid", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X2 = conv2d_bn(X, 64, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X2 = conv2d_bn(X2, 64, (7, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X2 = conv2d_bn(X2, 64, (1, 7), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X2 = conv2d_bn(X2, 96, (3, 3), strides=1, padding="valid", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X = concatenate([X1, X2], axis=3)

    X1 = conv2d_bn(X, 192, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X2 = MaxPool2D((3, 3), strides=(2, 2), padding="valid")(X)

    X = concatenate([X1, X2], axis=3, name=name)

    return X


def Inception_V4_A(X, kernel_initializer=None, bias_initializer=None, name=None):
    X1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(X)
    X1 = conv2d_bn(X1, 96, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X2 = conv2d_bn(X, 96, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X3 = conv2d_bn(X, 64, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X3 = conv2d_bn(X3, 96, (3, 3), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X4 = conv2d_bn(X, 64, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X4 = conv2d_bn(X4, 96, (3, 3), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X4 = conv2d_bn(X4, 96, (3, 3), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X = concatenate([X1, X2, X3, X4], axis=3, name=name)

    return X


def Inception_V4_B(X, kernel_initializer=None, bias_initializer=None, name=None):
    X1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(X)
    X1 = conv2d_bn(X1, 128, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X2 = conv2d_bn(X, 384, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X3 = conv2d_bn(X, 64, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X3 = conv2d_bn(X3, 96, (3, 3), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X4 = conv2d_bn(X, 64, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X4 = conv2d_bn(X4, 96, (3, 3), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X4 = conv2d_bn(X4, 96, (3, 3), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X = concatenate([X1, X2, X3, X4], axis=3, name=name)

    return X


def Inception_V4_C(X, kernel_initializer=None, bias_initializer=None, name=None):
    X1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(X)
    X1 = conv2d_bn(X1, 256, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X2 = conv2d_bn(X, 256, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X3 = conv2d_bn(X, 384, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X3_1 = conv2d_bn(X3, 256, (1, 3), activation="relu", kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)
    X3_2 = conv2d_bn(X3, 256, (3, 1), activation="relu", kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)

    X4 = conv2d_bn(X, 384, (1, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X4 = conv2d_bn(X4, 448, (1, 3), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X4 = conv2d_bn(X4, 512, (3, 1), activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X4_1 = conv2d_bn(X4, 256, (3, 1), activation="relu", kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)
    X4_2 = conv2d_bn(X4, 256, (1, 3), activation="relu", kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)

    X = concatenate([X1, X2, X3_1, X3_2, X4_1, X4_2], axis=3, name=name)

    return X


def Reduction_A(X, kernel_initializer=None, bias_initializer=None, name=None):
    X1 = MaxPool2D((3, 3), strides=(2, 2), padding="valid")(X)

    X2 = conv2d_bn(X, 384, (3, 3), strides=2, padding="valid", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X3 = conv2d_bn(X, 192, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X3 = conv2d_bn(X3, 224, (3, 3), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X3 = conv2d_bn(X3, 256, (3, 3), strides=2, padding="valid", activation="relu",
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X = concatenate([X1, X2, X3], axis=3, name=name)

    return X


def Reduction_B(X, kernel_initializer=None, bias_initializer=None, name=None):
    X1 = MaxPool2D((3, 3), strides=(2, 2), padding="valid")(X)

    X2 = conv2d_bn(X, 192, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X2 = conv2d_bn(X2, 192, (3, 3), strides=2, padding="valid", activation="relu",
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X3 = conv2d_bn(X, 256, (1, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X3 = conv2d_bn(X3, 256, (1, 7), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X3 = conv2d_bn(X3, 320, (7, 1), strides=1, padding="same", activation="relu", kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)
    X3 = conv2d_bn(X3, 320, (3, 3), strides=2, padding="valid", activation="relu",
                   kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)

    X = concatenate([X1, X2, X3], axis=3, name=name)

    return X


if __name__ == '__main__':
    pass