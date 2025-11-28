import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import torch

from src.agent.action_masking.action_masking_utils import (
    MaskContext,
    get_logit_slices,
    max_sampling,
    transform_decisions_to_action_indices,
    standard_stochastic_sampling,
)
from src.environment.environment import MAX_LAYERS
from src.agent.action_masking.action_masking import (
    sample_actions,
)
from src.utils.network_config import (
    get_number_of_actions_from_observation,
)
from src.agent.action_masking.action_masking_utils import EMPTY_DECISIONS
from src.utils.data_importer.cifar.cifar10 import DEFAULT_W, DEFAULT_H, NUM_CHANNELS


class CustomMaskablePolicy(MaskableActorCriticPolicy):
    """
    Custom Maskable Policy for PPO that integrates sequential action masking
    based on the current environment state. Supports single-observation forward pass.
    """

    # argument action_masks needs to be included to fulfill interface requirements.
    def forward(self, obs, deterministic: bool = False, action_masks=None):
        # 1. Feature extraction
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(
            features
        )  # pi is policy network features and vf is value network features

        # 2. Get raw policy network logits from features
        # Defensive: sanitize logits to avoid NaN/Inf creeping into downstream masking
        # (unstable training can produce NaNs which will crash distribution creation)
        logits = self.action_net(latent_pi).squeeze(0)  # [total_action_dim]
        # Convert any NaN/inf to large negative/positive numbers here so masking
        # and later distribution creation do not explode with NaNs.
        # If a NaN is present, we also log it so the training/debugger can pick it up.
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Log a little warning. Keep this lightweight so it won't spam too much.
            try:
                print("[warning] Detected NaN/Inf in policy logits — sanitizing (clamped).")
            except Exception:
                pass
        logits = torch.nan_to_num(logits, nan=-1e8, posinf=1e8, neginf=-1e8)

        sampling_strategy = standard_stochastic_sampling if not deterministic else max_sampling
        action_count = get_number_of_actions_from_observation(obs.squeeze(0).cpu().numpy())
        ctx = MaskContext(
            logits=logits.detach().cpu().numpy(),  # ensure format
            observation=obs.squeeze(0).cpu().numpy(),
            slices=get_logit_slices(MAX_LAYERS),
            sampling_strategy=sampling_strategy,
            max_layers=MAX_LAYERS,
            decisions=EMPTY_DECISIONS,
            input_dimensions=(NUM_CHANNELS, DEFAULT_H, DEFAULT_W),
            action_count=action_count,
        )

        # 4. Sample sequential actions
        # Here, decisions is an array where the first index corresponds to the action chosen for the first category, etc.
        decisions, masked_logits = sample_actions(ctx)

        # 5. Transform decisions to action indices tensor
        # Actions must be a tensor of a specific shape for SB3
        actions = torch.tensor(
            np.array(
                transform_decisions_to_action_indices(decisions, ctx.slices, ctx.max_layers),
                dtype=np.int64,
            ),
            device=obs.device,
            dtype=torch.long,
        ).unsqueeze(0)

        # 6. Compute log probability across all action categories, and value estimate from critic
        log_prob = self.compute_log_prob_from_masked_logits(masked_logits, decisions, ctx.slices)
        value = self.value_net(latent_vf)

        # must return actions, value estimate and log probability for policy.
        return actions, value, log_prob

    def compute_log_prob_from_masked_logits(self, masked_logits, decisions, slices):
        """
        Computes the total log probability of the selected actions given the masked logits.

        """
        category_names = [
            "standard_actions",
            "layer_type",
            "out_channels",
            "kernel_size",
            "stride",
            "linear_units",
            "pool_mode",
            "activation_function",
            "skip_connection",
        ]

        decision_values = [
            decisions.action_choice.value,
            decisions.layer_type_choice.value,
            decisions.out_channels_choice.value,
            decisions.kernel_size_choice.value,
            decisions.stride_choice.value,
            decisions.linear_units_choice.value,
            decisions.pool_mode_choice.value,
            decisions.activation_function_choice.value,
            decisions.skip_connection_choice if decisions.skip_connection_choice is not None else 0,
        ]

        log_probs = []
        for idx, cat in enumerate(category_names):
            logits_slice = masked_logits[slices.__dict__[cat].all].copy()
            # Replace NaN and -inf with large negative numbers — PyTorch distributions
            # don't tolerate NaN values and behave poorly with -inf only.
            logits_slice = np.nan_to_num(logits_slice, nan=-1e8, posinf=1e8, neginf=-1e8)
            logits_slice = torch.tensor(logits_slice, dtype=torch.float64)
            dist = torch.distributions.Categorical(logits=logits_slice)
            log_probs.append(dist.log_prob(torch.tensor(decision_values[idx], dtype=torch.long)))

        total_log_prob = torch.sum(torch.stack(log_probs))
        return total_log_prob

    # NOTE: Stable-baselines training flow sometimes calls `_get_action_dist_from_latent`
    # (from MaskableActorCriticPolicy) which builds a MaskableDistribution directly
    # from action logits. That path bypasses our `forward(...)` sanitization. To make
    # training safe we override that construction point and sanitize there as well.
    def _get_action_dist_from_latent(self, latent_pi):
        # call action_net like the base class does and ensure logits are safe
        action_logits = self.action_net(latent_pi)
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            try:
                print(
                    "[warning] Detected NaN/Inf in action_logits inside _get_action_dist_from_latent — sanitizing."
                )
            except Exception:
                pass
        action_logits = torch.nan_to_num(action_logits, nan=-1e8, posinf=1e8, neginf=-1e8)
        return self.action_dist.proba_distribution(action_logits=action_logits)
