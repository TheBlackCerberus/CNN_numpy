from functools import cache

import numpy as np

class Convolution:
    def __init__(self, filter_shape=(int, int), stride=2, padding='valid', pad=1):
        self.hparams = {
            "filter_shape" : filter_shape,
            "stride" : stride,
            "pad" : pad,
            "padding" : padding
        }
        self.cache = {}
        self.gradients = {}



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


    def forward_pass(self, out_activation, weights, biases):

        (dim_train, dim_height_prev, dim_width_prev, dim_channels_prev) = out_activation.shape
        (f, f, dim_channels_prev, dim_channels) = weights.shape
        pad = self.hparams["pad"]
        stride = self.hparams["stride"]

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

        out_layer = np.zeros((dim_train, dim_height,  dim_width, dim_channels))
        out_prev_A_pad = self.pad_by_zero(out_activation)

        for i in range(dim_train):
            out_prev_a = out_prev_A_pad[i]
            for h in range(dim_height):
                for w in range(dim_width):
                    for c in range(dim_channels):
                        height_start = h*stride
                        height_end = h*stride + f
                        width_start = w*stride
                        width_end = w*stride + f

                        out_slice_prev_a = out_prev_a[height_start:height_end, width_start:width_end,:]
                        out_layer[i,h, w, c] = self.single_convolution(out_slice_prev_a, weights[:, :, :, c], biases[:, :, :, c])

                        # add activation function
                        #A[i, h, w, c] = activation(Z[i, h, w, c])
        assert out_layer.shape == (dim_train, dim_height,  dim_width, dim_channels)

        # save the params for backprop
        self.cache["prev_activation"] = out_activation
        self.cache["weights"] = weights
        self.cache["biases"] = biases

        return out_layer

    def initialise_cache(self):
        cache = dict()
        cache["dW"] = np.zeros_like(self.cache["weights"])
        cache["db"] = np.zeros_like(self.cache["biases"])
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
                    for c in range(dim_channels):
                        height_start = h*stride
                        height_end = h*stride + f
                        width_start = w*stride
                        width_end = w*stride + f

                        a_slice = a_prev_pad[height_start:height_end, width_start:width_end, :]
                        da_prev_pad[height_start:height_end, width_start:width_end, :] += self.cache["weights"][:,:,:,c] * dZ[i,h,w,c]
                        self.gradients["dW"][:,:,:, c] += a_slice * dZ[i,h,w,c]
                        self.gradients["db"][:,:,:,c] += dZ[i, h, w, c]

            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

        assert(dA_prev.shape == (dim_train, dim_height_prev, dim_height_prev, dim_channels_prev))
        return dA_prev





