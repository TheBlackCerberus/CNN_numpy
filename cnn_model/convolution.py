from functools import cache

from cnn_numpy.cnn_model import initializers
#from initializers import glorot_uniform, glorot_normal, he_uniform, he_normal, random_normal

import numpy as np

class Convolution:
    def __init__(self, num_filters=1, filter_shape=(int, int), stride=2, padding='valid', pad=1, initializer="random_normal"):
        self.hparams = {
            "num_filters": num_filters,
            "filter_shape" : filter_shape,
            "stride" : stride,
            "pad" : pad,
            "padding" : padding,
            "initializer" : initializer
        }
        self.cache = {}
        self.gradients = {}

    def set_initialization(self, shape):
        if self.hparams["initializer"] == "he_uniform":
            W, b = initializers.he_uniform(shape)
        elif self.hparams["initializer"] == "he_normal":
            W, b = initializers.he_normal(shape)
        elif self.hparams["initializer"] == "glorot_uniform":
            W, b = initializers.glorot_uniform(shape)
        elif self.hparams["initializer"] == "glorot_normal":
            W, b = initializers.glorot_normal(shape)
        else:
            W, b = initializers.random_normal()
        return W, b

    def single_convolution(self, input_slice, weights, bias):
        """

        :param input_slice: shape => (f, f, dim_channels_prev)
        :param weights: shape => (f, f, dim_channels_prev)
        :param bias: shape => (1,1,1)
        :return:
        """
        scalar = np.sum(np.multiply(input_slice, weights)) + float(bias)
        return scalar

    def pad_by_zero(self, input_layer):
        # pad input layer with the shape dim_train, dim_height_prev, dim_width_prev, dim_channels_prev
        pad = self.hparams["pad"]
        return np.pad(input_layer, ((0,0),(pad,pad),(pad, pad),(0,0)), 'constant', constant_values=(0,0))


    def forward_pass(self, out_activation, save_cache=True):

        pad = self.hparams["pad"]
        stride = self.hparams["stride"]
        num_filters = self.hparams["num_filters"]
        f, f = self.hparams["filter_shape"]

        (dim_train, dim_height_prev, dim_width_prev, dim_channels_prev) = out_activation.shape
        if "W" not in self.cache:
            shape = (f, f, dim_channels_prev, num_filters)
            self.cache["W"], self.cache["b"] = self.set_initialization(shape)

        W, b = self.cache["W"], self.cache["b"]
        (_, _, dim_channels_prev, dim_channels) = W.shape

        # select the type of padding
        if self.hparams["padding"] == "same":
            dim_height = int((dim_height_prev - f + 2 * pad) / stride) + 1
            dim_width = int((dim_width_prev - f + 2 * pad) / stride) + 1
        elif self.hparams["padding"] == "valid":
            dim_height = int(dim_height_prev - f + 1)
            dim_width = int(dim_width_prev - f + 1)
        else:
            dim_height = int((dim_height_prev - f + 2 * pad) / stride) + 1
            dim_width = int((dim_width_prev - f + 2 * pad) / stride) + 1

        out_layer = np.zeros((dim_train, dim_height,  dim_width, num_filters))
        out_prev_A_pad = self.pad_by_zero(out_activation)

        for i in range(dim_train):
            out_prev_a = out_prev_A_pad[i]
            for h in range(dim_height):
                for w in range(dim_width):
                    for c in range(num_filters):
                        height_start = h*stride
                        height_end = h*stride + f
                        width_start = w*stride
                        width_end = w*stride + f

                        out_slice_prev_a = out_prev_a[height_start:height_end, width_start:width_end,:]
                        out_layer[i,h, w, c] = self.single_convolution(out_slice_prev_a, W[:, :, :, c], b[:, :, :, c])

                        # add activation function
                        #A[i, h, w, c] = activation(Z[i, h, w, c])
        print(out_layer.shape)
        assert out_layer.shape == (dim_train, dim_height,  dim_width, num_filters)

        # save the params for backprop
        self.cache["prev_activation"] = out_activation

        return out_layer

    def initialise_cache(self):
        cache = dict()
        cache["dW"] = np.zeros_like(self.cache["W"])
        cache["db"] = np.zeros_like(self.cache["b"])
        return cache

    def backward_pass(self, dZ):
        """

        :param dZ: gradient with respect to output of CNN layer => shape (dim_train, dim_height, dim_width, dim_channels)
        :return: return gradient with respect to previous CNN layer
        """

        prev_activation = self.cache["prev_activation"]
        (dim_train, dim_height_prev, dim_height_prev, dim_channels_prev) = prev_activation.shape
        (f, f) = self.hparams["filter_shape"]
        stride = self.hparams["stride"]
        pad = self.hparams["pad"]
        num_filters = self.hparams["num_filters"]

        (dim_train, dim_height, dim_width, dim_channels) = dZ.shape

        dA_prev = np.zeros((dim_train, dim_height_prev, dim_height_prev, dim_channels_prev))
        self.gradients = self.initialise_cache()

        A_prev_pad = self.pad_by_zero(prev_activation)
        dA_prev_pad = self.pad_by_zero(dA_prev)

        for i in range(dim_train):

            for h in range(dim_height):
                a_prev_pad = A_prev_pad[i,...]
                da_prev_pad = dA_prev_pad[i,...]
                for w in range(dim_width):
                    for c in range(num_filters):
                        height_start = h*stride
                        height_end = h*stride + f
                        width_start = w*stride
                        width_end = w*stride + f

                        a_slice = a_prev_pad[height_start:height_end, width_start:width_end, :]
                        da_prev_pad[height_start:height_end, width_start:width_end, :] += self.cache["W"][:,:,:,c] * dZ[i,h,w,c]
                        self.gradients["dW"][:,:,:, c] += a_slice * dZ[i,h,w,c]
                        self.gradients["db"][:,:,:,c] += dZ[i, h, w, c]

            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

        assert(dA_prev.shape == (dim_train, dim_height_prev, dim_height_prev, dim_channels_prev))
        return dA_prev





