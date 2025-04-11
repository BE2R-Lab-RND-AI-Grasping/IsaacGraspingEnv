import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.utils.spaces.torch import flatten_tensorized_space, unflatten_tensorized_space

# define the model
class PCPolicy(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, feature_extractor,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, role="policy")
        DeterministicMixin.__init__(self, clip_actions, role="value")

        self.state_dim = self.num_observations - np.prod(observation_space["point_cloud"].shape)
        
        self.feature_extraction = feature_extractor
        
        self.value_nn = nn.Sequential(
                nn.Linear(self.feature_extraction.n_output_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        
        self.pi_mean_nn = nn.Sequential(
                nn.Linear(self.feature_extraction.n_output_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_actions),
            )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
        
    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        
        obs = unflatten_tensorized_space(self.observation_space, inputs["states"])

        features = self.feature_extraction(obs)

        if role == "policy":
            # save shared layers/network output to perform a single forward-pass
            return self.pi_mean_nn(features), self.log_std_parameter, {}
        elif role == "value":
            # use saved shared layers/network output to perform a single forward-pass, if it was saved
            return self.value_nn(features), {}