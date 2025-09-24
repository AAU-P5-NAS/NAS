CNN_ACTION_SPACE = {
    "layer_type": ["conv", "pool", "linear", "stop"],
    "out_channels": [16, 32, 64, 128],
    "kernel_size": [1, 3, 5],
    "stride": [1, 2],
    "pool_mode": ["max", "avg"],
    "activation": ["relu", "tanh", "none"],
    "linear_units": [64, 128, 256, 512],
}
