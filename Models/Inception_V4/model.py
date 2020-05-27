from Models.Inception_V4.blocks import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, Activation, BatchNormalization, Add, AveragePooling2D, Input, ZeroPadding2D, concatenate, GlobalAveragePooling2D, Lambda


def Inception_V4(size=(299, 299, 3), N_Classes=5, kernel_initializer=None, bias_initializer=None):
    input_layer = Input(shape=size)

    X = Stem(input_layer, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Stem")

    X = Inception_V4_A(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_A_1")
    X = Inception_V4_A(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_A_2")
    X = Inception_V4_A(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_A_3")
    X = Inception_V4_A(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_A_4")

    X = Reduction_A(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Reduction_A")

    X = Inception_V4_B(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_B_1")
    X = Inception_V4_B(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_B_2")
    X = Inception_V4_B(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_B_3")
    X = Inception_V4_B(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_B_4")
    X = Inception_V4_B(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_B_5")
    X = Inception_V4_B(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_B_6")
    X = Inception_V4_B(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_B_7")

    X = Reduction_B(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Reduction_B")

    X = Inception_V4_C(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_C_1")
    X = Inception_V4_C(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_C_2")
    X = Inception_V4_C(X, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Block_C_3")

    X = GlobalAveragePooling2D(name="Global_Average_Pooling")(X)

    X = Dropout(.8)(X)

    X = Dense(N_Classes, activation="softmax", name="final_output")(X)

    model = Model(input_layer, X, name="Inception_V4")
    return model


if __name__ == '__main__':
    pass