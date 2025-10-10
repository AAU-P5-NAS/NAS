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
    InvalidLayerConfigError,
)

from src.utils.cnn_builder import CNNBuilder


class InvalidArchitectureAction(ValueError):
    pass


def arch_builder(actions: list[int], partial_arch: NetworkConfig) -> NetworkConfig:
    """
    Input: Takes a list of action and partialy build arch

    Output: Return the partually build with the new layer appended to it

    Note:
    the actions must be in the following order, otherwise, the method will fail when calling .build() on the constructed Network
    [action, layerIdx, layerType, outCh, kernelSize, stride, linearU,  poolMode, actFun]

    Also, currently it only append the layer at the end.

    """
    if actions[0] == 0:
        # no oberation
        return partial_arch
    if actions[0] == 1:
        # remove layer
        pass
    elif actions[0] == 2:
        # modify layer
        pass
    elif actions[0] == 3:
        # add layer
        return add_layer(actions, partial_arch)


def remove_layer(actions: list[int], partial_arch: NetworkConfig):
    raise NotImplementedError


def modify_layer(actions: list[int], partial_arch: NetworkConfig):
    raise NotImplementedError


def add_layer(actions: list[int], partial_arch: NetworkConfig):
    try:
        lt = LayerType(actions[2])
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
    except ValueError:
        raise InvalidArchitectureAction("Cannot add a layer of type None")

    partial_arch.layers.append(layerConfig)

    layer_idx = actions[1]
    print("layer_idx", layer_idx)
    print("len(partial_arch.layers)", len(partial_arch.layers))
    if layer_idx == len(partial_arch.layers):
        partial_arch.layers.append(layerConfig)
        return partial_arch
    else:
        partial_arch.layers.insert(layer_idx, layerConfig)
        return partial_arch
        if check_compatibility(partial_arch):
            return partial_arch
        else:
            raise InvalidLayerConfigError("The partial arch is not compatible")


def check_compatibility(partial_arch) -> bool:
    raise NotImplementedError


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
        ]
    )

    for layer in config.layers:
        print(layer)
    cnn_builder = CNNBuilder(rl_config=config, input_size=(28, 28), num_classes=26)
    model = cnn_builder.build()
    # Print the model architecture

    actions = [1, 2, 1, 0, 0, 0, 3, 0, 1]
    action1 = [1, 1, 2, 0, 1, 0, 0, 1, 3]
    arch_builder(action1, config)

    for layer in config.layers:
        cnn_builder = CNNBuilder(rl_config=config, input_size=(28, 28), num_classes=26)
        model = cnn_builder.build()
        # Print the model architecture
