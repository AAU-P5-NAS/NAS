import src.model_module.environment as environment_module
from src.classification_module.reward import WeightedSumRS


def get_environment():
    environment = environment_module.CustomEnv(
        logdir="tests/logs",
        device="cpu",
        reward_strategy=WeightedSumRS(),
    )
    return environment


def test_environment_initialization():
    environment = get_environment()
    assert environment.logdir == "tests/logs"
    assert environment.device == "cpu"
    assert isinstance(environment.reward_strategy, WeightedSumRS)
    assert environment.max_layers == 16
    assert environment.training_epochs == 15
    assert environment.data_importer is not None


def test_get_new_architecture():
    import numpy as np
    import src.utils.network_utils as nu

    environment = get_environment()

    action_logits = [
        # Standard Action
        0,  # None
        1,  # Add layer
        # Layer Type
        0,  # None
        1,  # Conv
        0,  # Linear
        0,  # Pool
        # Out Channels
        0,  # None
        0,  # CH_16
        0,  # CH_32
        0,  # CH_64
        0,  # CH_128
        0,  # CH_256
        # Kernel Size
        0,  # None
        0,  # KS_1
        1,  # KS_3
        0,  # KS_5
        # Stride
        0,  # None
        1,  # STR_1
        0,  # STR_2
        # Linear Units
        0,  # None
        0,  # LU_64
        0,  # LU_128
        0,  # LU_256
        0,  # LU_512
        # Pool Mode
        0,  # None
        0,  # MaxPool
        0,  # AvgPool
        # Activation Function
        1,  # None
        1,  # ReLU
        1,  # Tanh
        1,  # Softmax
    ]
    action_logits += [0] * environment.max_layers  # add skip logits

    architecture, should_evaluate = environment._get_new_architecture(
        action_logits=np.array(action_logits, dtype=np.float32)
    )
    assert isinstance(architecture, environment_module.NetworkConfig)
    assert isinstance(should_evaluate, bool)
    assert len(architecture.layers) == 1
    assert architecture.layers[0].activation == nu.ActivationFunction.RELU
    assert architecture.layers[0].layer_type == nu.LayerType.CONV
    assert architecture.layers[0].out_channels == nu.OutChannels.CH_16
    assert architecture.layers[0].kernel_size == nu.KernelSize.KS_3
    assert architecture.layers[0].stride == nu.Stride.S_1
    assert should_evaluate is False

    action_logits = [
        # Standard Action
        0,  # None
        1,  # Add layer
        # Layer Type
        0,  # None
        0,  # Conv
        1,  # Linear
        0,  # Pool
        # Out Channels
        0,  # None
        0,  # CH_16
        1,  # CH_32
        0,  # CH_64
        0,  # CH_128
        0,  # CH_256
        # Kernel Size
        0,  # None
        0,  # KS_1
        0,  # KS_3
        0,  # KS_5
        # Stride
        0,  # None
        0,  # STR_1
        0,  # STR_2
        # Linear Units
        0,  # None
        0,  # LU_64
        0,  # LU_128
        1,  # LU_256
        0,  # LU_512
        # Pool Mode
        0,  # None
        0,  # MaxPool
        0,  # AvgPool
        # Activation Function
        0,  # None
        1,  # ReLU
        0,  # Tanh
        0,  # Softmax
    ]
    action_logits += [0] * environment.max_layers  # add skip logits

    architecture, should_evaluate = environment._get_new_architecture(
        action_logits=np.array(action_logits, dtype=np.float32)
    )
    assert isinstance(architecture, environment_module.NetworkConfig)
    assert isinstance(should_evaluate, bool)
    assert len(architecture.layers) == 2
    assert architecture.layers[1].activation == nu.ActivationFunction.RELU
    assert architecture.layers[1].layer_type == nu.LayerType.LINEAR
    assert architecture.layers[1].linear_units == nu.LinearUnits.LU_256
    assert should_evaluate is False

    action_logits = [
        # Standard Action
        1,  # None
        0,  # Add layer
        # Layer Type
        0,  # None
        0,  # Conv
        1,  # Linear
        0,  # Pool
        # Out Channels
        0,  # None
        0,  # CH_16
        1,  # CH_32
        0,  # CH_64
        0,  # CH_128
        0,  # CH_256
        # Kernel Size
        0,  # None
        0,  # KS_1
        0,  # KS_3
        0,  # KS_5
        # Stride
        0,  # None
        0,  # STR_1
        0,  # STR_2
        # Linear Units
        0,  # None
        0,  # LU_64
        0,  # LU_128
        1,  # LU_256
        0,  # LU_512
        # Pool Mode
        0,  # None
        0,  # MaxPool
        0,  # AvgPool
        # Activation Function
        0,  # None
        1,  # ReLU
        0,  # Tanh
        0,  # Softmax
    ]
    action_logits += [0] * environment.max_layers  # add skip logits

    architecture, should_evaluate = environment._get_new_architecture(
        action_logits=np.array(action_logits, dtype=np.float32)
    )
    assert isinstance(architecture, environment_module.NetworkConfig)
    assert isinstance(should_evaluate, bool)
    assert len(architecture.layers) == 2
    assert should_evaluate is True
