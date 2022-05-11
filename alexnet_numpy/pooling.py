import numpy as np

class Pooling:
    def __init__(self, filter_shape=(int,int), mode='mean', stride=1):
        self.hparams = {
            "filter_shape" : filter_shape,
            "stride" : stride,
        }
        self.mode = mode
        self.cache = {}

    def forward_pass(self, input_prev, save_cache=True):
        (dim_train, dim_height_prev, dim_width_prev, dim_channels_prev) = input_prev.shape
        (f, f) = self.hparams["filter_shape"]
        stride = self.hparams["stride"]

        dim_height = int((dim_height_prev - f)/stride) + 1
        dim_width = int((dim_width_prev - f)/stride) + 1
        dim_channels = dim_channels_prev

        out_pool_layer = np.zeros((dim_train, dim_height, dim_width, dim_channels))

        for i in range(dim_train):
            for h in range(dim_height):
                for w in range(dim_width):
                    for c in range(dim_channels):
                        height_start = h*stride
                        height_end = h*stride + f
                        width_start = w*stride
                        width_end = w*stride + f

                        input_prev_slice = input_prev[i, height_start:height_end, width_start:width_end, c]

                        if self.mode == "mean":
                            out_pool_layer[i,h,w,c] = np.mean(input_prev_slice)
                        elif self.mode == "max":
                            out_pool_layer[i,h,w,c] = np.max(input_prev_slice)
                        elif self.mode == "min":
                            out_pool_layer[i,h,w,c] = np.min(input_prev_slice)
                        else:
                            out_pool_layer[i,h,w,c] = np.mean(input_prev_slice)

        self.cache["A"] = input_prev
        assert out_pool_layer.shape == (dim_train, dim_height, dim_width, dim_channels)
        return out_pool_layer

    def distribute_value(self, dz, shape):

        (dim_height, dim_width) = shape
        average = np.prod(shape)
        a = (dz/average)*np.ones(shape)
        return a

    def create_mask(self, x):
        if self.mode == "max":
            mask = x == np.max(x)
        elif self.mode == "min":
            mask = x == np.min(x)
        else:
            mask = x == np.max(x)
        return mask


    def backward_pass(self, dA):

        (f, f) = self.hparams["filter_shape"]
        stride = self.hparams["stride"]
        A_prev = self.cache["A"]

        dim_train, dim_height_prev, dim_width_prev, dim_channels_prev = A_prev.shape
        dim_train, dim_height, dim_width, dim_channels = dA.shape

        dA_prev = np.zeros(A_prev.shape)

        for i in range(dim_train):
            a_prev = A_prev[i,...]
            for h in range(dim_height):
                for w in range(dim_width):
                    for c in range(dim_channels):
                        height_start = h*stride
                        height_end = h*stride + f
                        width_start = w*stride
                        width_end = w*stride + f

                        if self.mode == "mean":
                            da = dA[i,h,w,c]
                            dA_prev[i,height_start:height_end,width_start:width_end,c] += self.distribute_value(da, self.hparams["filter_shape"])
                        elif self.mode == "max" or "min":
                            a_prev_slice = a_prev[height_start:height_end,width_start:width_end,c]
                            mask = self.create_mask(a_prev_slice)
                            dA_prev[i,height_start:height_end,width_start:width_end,c] += mask*dA[i,h,w,c]
                        else:
                            da = dA[i,h,w,c]
                            dA_prev[i,height_start:height_end,width_start:width_end,c] += self.distribute_value(da, self.hparams["filter_shape"])


        assert(dA_prev.shape == A_prev.shape)

        return dA_prev









