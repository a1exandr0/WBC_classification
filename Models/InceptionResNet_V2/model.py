from Models.InceptionResNet_V2.blocks import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten, Activation, BatchNormalization, Add, AveragePooling2D, Input, ZeroPadding2D, concatenate, GlobalAveragePooling2D, Lambda


def InceptionResNet_V2(size=(299, 299, 3), N_classes=5, kernel_init=None, bias_init=None):
    input_layer = Input(shape=size)

    X = Stem(input_layer, kernel_init, bias_init, name="Stem")

    X = InceptionResNet_A(X, kernel_init, bias_init, name="Block_A_1", scale=.15)
    X = InceptionResNet_A(X, kernel_init, bias_init, name="Block_A_2", scale=.15)
    X = InceptionResNet_A(X, kernel_init, bias_init, name="Block_A_3", scale=.15)
    X = InceptionResNet_A(X, kernel_init, bias_init, name="Block_A_4", scale=.15)
    X = InceptionResNet_A(X, kernel_init, bias_init, name="Block_A_5", scale=.15)

    X = Reduction_A(X, kernel_init, bias_init, name="Reduction_block_A")

    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_1", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_2", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_3", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_4", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_5", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_6", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_7", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_8", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_9", scale=.1)
    X = InceptionResNet_B(X, kernel_init, bias_init, name="Block_B_10", scale=.1)

    X = Reduction_B(X, kernel_init, bias_init, name="Reduction_block_B")

    X = InceptionResNet_C(X, kernel_init, bias_init, name="Block_C_1", scale=.2)
    X = InceptionResNet_C(X, kernel_init, bias_init, name="Block_C_2", scale=.2)
    X = InceptionResNet_C(X, kernel_init, bias_init, name="Block_C_3", scale=.2)

    X = GlobalAveragePooling2D(name="GlobalAvgPool")(X)

    X = Dropout(.8)(X)

    X = Dense(N_classes, activation="softmax", name="final_output")(X)

    model = Model(input_layer, X, name="InceptionResNet_V2")

    return model

if __name__ == '__main__':
    nn = InceptionResNet_V2()
