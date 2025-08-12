
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


class RewardAnalyzer:
    
    def __init__(self, reward_manager, episode_length, num_envs = 1) -> None:
        
        self.reward_manager = reward_manager
        
        self._env = self.reward_manager._env
        self.term_names = self.reward_manager._term_names
        self._term_cfgs = self.reward_manager._term_cfgs
        self.episode_length = episode_length
        self.num_envs = num_envs
        
        self.ep_history = {name:[[] for __ in range(num_envs)] for name in self.term_names}
        self.ep_history["reward_wo_weight"] = [[] for __ in range(self.num_envs)]
        self.ep_history["reward_with_weight"] = [[] for __ in range(self.num_envs)]
        
        self.term_history = {name:[] for name in self.term_names}
        self.term_history["reward_wo_weight"] = []
        self.term_history["reward_with_weight"] = []
        
    def step_update(self):
        
        for term_idx, (name, term_cfg) in enumerate(zip(self.term_names, self._term_cfgs)):
            # compute term's value
            step_reward = term_cfg.func(self._env, **term_cfg.params).cpu().numpy()
            for i in range(self.num_envs):
                if term_idx == 0:
                    self.ep_history["reward_wo_weight"][i].append(step_reward[i])
                    self.ep_history["reward_with_weight"][i].append(step_reward[i]*term_cfg.weight)
                else:
                    self.ep_history["reward_wo_weight"][i][-1] = self.ep_history["reward_wo_weight"][i][-1] + step_reward[i]
                    self.ep_history["reward_with_weight"][i][-1] = self.ep_history["reward_with_weight"][i][-1] + step_reward[i]*term_cfg.weight
                
                self.ep_history[name][i].append(step_reward[i])
            
    def episode_update(self, id_env=0):

        for name in self.term_history.keys():
            if len(self.ep_history[name][id_env]) == self.episode_length:
                self.term_history[name].append(self.ep_history[name][id_env])
            self.ep_history[name][id_env] = []
            
    
    def plot(self, episode: Optional[int] = None):
        
        terms_name = self.term_history.keys()
        n = len(terms_name)  # number of plots
        cols = np.ceil(np.sqrt(n))
        rows = np.ceil(n / cols)
        
        fig, axes = plt.subplots(int(rows), int(cols), figsize=(22, 15.4))
        axes = axes.flatten()
        term_weights = [cfg.weight for cfg in  self._term_cfgs]
        term_weights = term_weights + [0.0, 0.0]
        for ax, name, term_weight in zip(axes, terms_name, term_weights):
            num_samples = 1
            if episode:
                data = np.asarray(self.term_history[name][episode]).squeeze()
                ax.plot(np.arange(data.shape[-1]), data, linewidth=2)
            else:
                data = np.asarray(self.term_history[name]).squeeze()
                if len(data.shape) == 1:
                    data = data[np.newaxis, :]
                num_samples = data.shape[0]
                
                mean_data = np.mean(data, axis=0)
                std_data = np.std(data, axis=0)
                
                range_steps = np.arange(data.shape[-1])
                
                ax.plot(range_steps, mean_data, '-', linewidth=2)
                ax.fill_between(range_steps, mean_data - std_data, mean_data + std_data, alpha=0.2)
            
            fig.suptitle(f'Mean reward value {num_samples} samples')
            ax.set_title(f"{name} ep: {episode}; w: {np.round(term_weight, 3)}")
            ax.set_xlabel("Env Step")
            ax.set_ylabel("Value")
            ax.grid(True)
        
        for i in range(n, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

