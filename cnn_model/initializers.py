import numpy as np


def compute_fans(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out


def he_uniform(shape):
    fan_in, fan_out = compute_fans(shape)
    scale = np.sqrt(6. / fan_in)
    weight_shape = (fan_out, fan_in) if len(shape) == 2 else shape
    bias_shape = (fan_out, 1) if len(shape) == 2 else (1, 1, 1, shape[3])
    weight = np.random.uniform(low=-scale, high=scale, size=weight_shape)
    bias = np.random.uniform(low=-scale, high=scale, size=bias_shape)
    return weight,bias


def he_normal(shape):
    fan_in, fan_out = compute_fans(shape)
    scale = np.sqrt(2. / fan_in)
    weight_shape = (fan_out, fan_in) if len(shape) == 2 else shape
    bias_shape = (fan_out, 1) if len(shape) == 2 else (1, 1, 1, shape[3])
    weight = np.random.normal(loc=0.0,scale=scale,size=weight_shape)
    bias = np.random.uniform(low=-scale, high=scale, size=bias_shape)
    return weight, bias


def glorot_uniform(shape):
    fan_in, fan_out = compute_fans(shape)
    scale = np.sqrt(6. / (fan_in + fan_out))
    weight_shape = (fan_out, fan_in) if len(shape) == 2 else shape
    bias_shape = (fan_out, 1) if len(shape) == 2 else (1, 1, 1, shape[3])
    weight = np.random.uniform(low=-scale, high=scale, size=weight_shape)
    bias = np.random.uniform(low=-scale, high=scale, size=bias_shape)
    return weight, bias


def glorot_normal(shape):
    fan_in, fan_out = compute_fans(shape)
    scale = np.sqrt(2. / (fan_in + fan_out))
    weight_shape = (fan_out, fan_in) if len(shape) == 2 else shape
    bias_shape = (fan_out, 1) if len(shape) == 2 else (1, 1, 1, shape[3])
    weight = np.random.normal(loc=0.0,scale=scale,size=weight_shape)
    bias = np.random.uniform(low=-scale, high=scale, size=bias_shape)
    return weight, bias


def random_normal():
    # weight = np.random.randn(3, 3, 4, 8)
    # bias = np.random.randn(1, 1, 1, 8)
    weight = np.random.randn(2, 2, 3, 8)
    bias = np.random.randn(1, 1, 1, 8)
    return weight, bias

