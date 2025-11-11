from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch

from src.model_module.action_builder import standard_stochastic_sampling
from src.model_module.environment import MAX_LAYERS
from src.utils.action_builder_utils import (
    MaskContext,
    get_logit_slices,
    sample_actions,
    transform_decisions_to_action_indices,
)
from src.utils.network_utils import EMPTY_DECISIONS

from src.data_module.cifar.cifar10 import DEFAULT_W, DEFAULT_H, NUM_CHANNELS


class CustomMaskablePolicy(MaskableActorCriticPolicy):
    """
    Custom Maskable Policy for PPO that integrates action masking based on the environment's state.
    Extends the MaskableActorCriticPolicy from sb3-contrib to include custom action sampling logic.

    :How it works:
    - The `forward` method is overridden to implement custom logic for action selection.
    - It uses the `MaskContext` to manage logits, observations, and action sampling strategies.
    - it must return the actions chosen, the value estimate for the critic part, and the log probabilities of the actions for the policy gradient update.


    """

    def forward(self, obs, deterministic: bool = False):
        # extract_features converts the observation into a format suitable for the policy network
        features = self.extract_features(obs)

        # pass the features through the MLP extractor to get latent representations for policy and value networks
        # latent means the last hidden layer outputs, basically the outputs of the neural network.
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Get the logits from the policy network (must transform latent_pi representation to logits)
        logits = self.action_net(latent_pi)

        ctx = MaskContext(
            logits=logits,
            observation=obs,
            slices=get_logit_slices(),
            sampling_strategy=standard_stochastic_sampling,
            max_layers=MAX_LAYERS,
            decisions=EMPTY_DECISIONS,
            input_dimensions=(DEFAULT_W, DEFAULT_H, NUM_CHANNELS),
        )

        actions, masked_logits = sample_actions(ctx)
        # Transform the sampled actions into their corresponding indices within the logits array (Not a decisions object)
        actions_indices = transform_decisions_to_action_indices(actions, ctx.slices)

        # Create a categorical distribution over the masked logits to compute log probabilities
        dist = torch.distributions.Categorical(logits=torch.tensor(masked_logits))
        # Compute the log probability of the selected actions
        log_prob = dist.log_prob(torch.tensor(actions_indices))

        # Get the value estimate from the value network
        value = self.value_net(latent_vf)

        # Return the actions, value network result, and policy network result (log probabilities).
        return actions_indices, value, log_prob

    pass
