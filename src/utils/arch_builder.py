from src.utils.network_utils import (
    ActivationFunction,
    KernelSize,
    LayerConfig,
    LayerType,
    LinearUnits,
    NetworkConfig,
    OutChannels,
    PoolMode,
    Stride,
)

from src.utils.cnn_builder import CNNBuilder


def arch_builder(actions: list[int], partial_arch: NetworkConfig) -> NetworkConfig:
    """
    Input: Takes a list of action and partialy build arch

    Output: Return the partually build with the new layer appended to it

    Note:
    the actions must be in the following order, otherwise, the method will fail when calling .build() on the constructed Network
    [action, layerIdx, layerType, outCh, kernelSize, stride, linearU,  poolMode, actFun]

    Also, currently it only append the layer at the end so it makes no use of "action" and "layerIdx"

    """
    if actions[0] == 0:
        pass
    elif actions[0] == 1:
        pass
    elif actions[0] == 2:
        pass
    elif actions[0] == 3:
        pass

    try:
        lt = LayerType(actions[2])
    except ValueError:
        lt = None
    try:
        oc = OutChannels(actions[3])
    except ValueError:
        oc = None
    try:
        ks = KernelSize(actions[4])
    except ValueError:
        ks = None
    try:
        st = Stride(actions[5])
    except ValueError:
        st = None
    try:
        lu = LinearUnits(actions[6])
    except ValueError:
        lu = None
    try:
        pm = PoolMode(actions[7])
    except ValueError:
        pm = None
    try:
        act = ActivationFunction(actions[8])
    except ValueError:
        act = None

    layerConfig = LayerConfig(
        layer_type=lt,
        out_channels=oc,
        kernel_size=ks,
        stride=st,
        linear_units=lu,
        pool_mode=pm,
        activation=act,
    )

    partial_arch.layers.append(layerConfig)
    return partial_arch


def remove_layer():
    raise NotImplementedError


def modify_layer():
    raise NotImplementedError


def add_layer():
    raise NotImplementedError


# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    # Define a sample RL-generated config
    config = NetworkConfig(
        layers=[
            LayerConfig(
                layer_type=LayerType.CONV,
                out_channels=OutChannels.CH_16,
                kernel_size=KernelSize.KS_3,
                activation=ActivationFunction.RELU,
            ),
            LayerConfig(
                layer_type=LayerType.POOL, pool_mode=PoolMode.MAX, kernel_size=KernelSize.KS_1
            ),
            #     LayerConfig(
            #         layer_type=LayerType.LINEAR,
            #         linear_units=LinearUnits.LU_64,
            #         activation=ActivationFunction.TANH,
            #     ),
        ]
    )

    for layer in config.layers:
        print(layer)
    cnn_builder = CNNBuilder(rl_config=config, input_size=(28, 28), num_classes=26)
    model = cnn_builder.build()
    # Print the model architecture
    print("Built CNN model:")
    print(model)

    actions = [1, 2, 1, -1, -1, -1, 3, -1, 1]
    action1 = [1, 1, 2, -1, 0, -1, -1, 0, 3]
    arch_builder(action1, config)

    print("the new config")

    for layer in config.layers:
        print(layer)

    cnn_builder = CNNBuilder(rl_config=config, input_size=(28, 28), num_classes=26)
    model = cnn_builder.build()
    # Print the model architecture
    print("Built CNN model:")
    print(model)
