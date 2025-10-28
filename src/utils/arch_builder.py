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


class InvalidArchitectureAction(ValueError):
    pass


class ArchBuilder:
    def __init__(self):
        pass

    def extend(self, actions: list[int], partial_arch: NetworkConfig) -> NetworkConfig:
        """
        Input: Takes a list of action and partially builds architecture

        Output: Returns the partially built architecture with the new layer appended to it

        Note:
        The actions must be in the following order, otherwise, the method will fail when calling .build() on the constructed Network:
        [action, layerIdx, layerType, outCh, kernelSize, stride, linearU, poolMode, actFun]

        Currently, it only appends the layer at the end.
        """
        if actions[0] == 0:
            # no operation
            return partial_arch
        elif actions[0] == 1:
            # remove layer
            return self.remove_layer(actions)
        elif actions[0] == 2:
            # modify layer
            return self.modify_layer(actions)
        elif actions[0] == 3:
            # add layer
            return self.add_layer(actions, partial_arch)
        return partial_arch

    def remove_layer(self, actions: list[int]) -> NetworkConfig:
        raise NotImplementedError

    def modify_layer(self, actions: list[int]) -> NetworkConfig:
        raise NotImplementedError

    def add_layer(self, actions: list[int], partial_arch: NetworkConfig) -> NetworkConfig:
        lt = LayerType(actions[2])
        oc = OutChannels(actions[3])
        ks = KernelSize(actions[4])
        st = Stride(actions[5])
        lu = LinearUnits(actions[6])
        pm = PoolMode(actions[7])
        act = ActivationFunction(actions[8])

        layer_config = LayerConfig(
            layer_type=lt,
            out_channels=oc,
            kernel_size=ks,
            stride=st,
            linear_units=lu,
            pool_mode=pm,
            activation=act,
        )
        layer_idx = actions[1]
        if layer_idx == len(partial_arch.layers):
            partial_arch.layers.append(layer_config)
        else:
            partial_arch.layers.insert(layer_idx, layer_config)

        return partial_arch


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

    arch_builder = ArchBuilder()
    arch_builder.extend(action1, config)

    for layer in config.layers:
        cnn_builder = CNNBuilder(rl_config=config, input_size=(28, 28), num_classes=26)
        model = cnn_builder.build()
        # Print the model architecture
