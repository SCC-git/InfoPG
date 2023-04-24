from typing import Optional, Dict, List
from alex_net import Encoder
from run_funcs import create_policies_from_experiment
from policy_piston import PistonPolicy
from policy_base import Experience
from batch_piston_env import PistonEnv
from torch.distributions import Categorical
import torch
import torch.optim as optim
import numpy as np
import random
import os
import pickle
import datetime
import json

class Piston5AgentByz(PistonEnv):
    def __init__(self, batch: int, env_params: Dict, seed=None):
        super().__init__(batch, env_params, seed=seed)

    def loop(self, user_params: Dict, policies: Dict[str, PistonPolicy], optimizers: Dict[str, torch.optim.Optimizer], schedulers):
        """
        Executes training loop on batch environment

        :param user_params: dict of user params
        :param policies: dict of policies, mapping agent_name(str) -> Policy()
        :param optimizers: dict of optimizers, mapping agent_name(str) -> Optimizer()
        :return: policies, optimizers, summary_stats
        """

        device = user_params['device']
        alex_encoder = Encoder(device)
        epochs = user_params['epochs']
        verbose = user_params['verbose']
        communicate = user_params['communicate']
        max_grad_norm = user_params['max_grad_norm']
        time_penalty = user_params['time_penalty']
        early_reward_benefit = user_params['early_reward_benefit']
        k_levels = user_params['k-levels']

        initial_params = {}
        for agent in self.AGENT_NAMES:
            initial_params[agent] = policies[agent].policy.state_dict().__str__()

        if 'consensus_update' in user_params.keys():
            consensus_update = user_params['consensus_update']
        else:
            consensus_update = False
        summary_stats = []
        for epoch in range(0, epochs):
            if verbose:
                print('Epoch: %s' % (epoch))
            observations = self.batch_reset()
            for step in range(self.MAX_CYCLES):
                num_left_batches = np.count_nonzero(self.DONE_ENVS == False)  # number of environments that aren't done
                left_batches = np.where(self.DONE_ENVS == False)[0]  # indeces of batches that aren't done (length is num_left_batches)
                if num_left_batches == 0:
                    break

                memory = {}  # this is a map of agent_name -> a list of length self.BATCH_SIZE, but batches that are done are set to None
                for agent in self.AGENT_NAMES:
                    memory[agent] = []
                    for actual_batch_ix in range(0, self.BATCH_SIZE):
                        if not self.DONE_ENVS[actual_batch_ix]:
                            memory[agent].append(Experience())
                        else:
                            memory[agent].append(None)
                    assert len(memory[agent]) == self.BATCH_SIZE, 'Error Here'

                with torch.no_grad():
                    for agent in self.AGENT_NAMES:
                        observations[agent] = alex_encoder(observations[agent])

                policy_initial = {}
                for agent in self.AGENT_NAMES:
                    initial_policy_distribution, state_val = policies[agent].forward(observations[agent], 0, None)
                    for batch_ix in range(0, num_left_batches):
                        memory[agent][left_batches[batch_ix]].state_val = state_val[batch_ix]
                    policy_initial[agent] = initial_policy_distribution

                for k in range(0, k_levels):
                    output_dist = {}
                    for agent_ix, agent in enumerate(self.AGENT_NAMES):
                        if communicate:
                            batched_neighbors = [[] for _ in range(0, num_left_batches)] # for each batch, the policies of agent
                            for batch_ix in range(0, num_left_batches):
                                actual_batch_number = left_batches[batch_ix]
                                neighbor_mask_for_agent = (self.adj_matrix[actual_batch_number][agent_ix] == 1)
                                neighbor_names = self.AGENT_NAMES[neighbor_mask_for_agent]
                                for neighbor in neighbor_names:
                                    batched_neighbors[batch_ix].append([policy_initial[neighbor][batch_ix], neighbor, batch_ix])
                            latent_vector = policies[agent].forward(policy_initial[agent], 1, batched_neighbors)
                        else:
                            latent_vector = policy_initial[agent]
                        output_dist[agent] = latent_vector
                    policy_initial = output_dist

                actions = {agent_name: [-1 for _ in range(0, num_left_batches)] for agent_name in self.AGENT_NAMES}
                for agent in self.AGENT_NAMES:
                    final_policy_distribution = policies[agent].forward(policy_initial[agent], 2, None)
                    final_policy_distribution = final_policy_distribution.to('cpu')
                    distribution = Categorical(probs=final_policy_distribution)
                    batch_action = distribution.sample()
                    batched_log_prob = distribution.log_prob(batch_action)

                    for batch_ix in range(0, num_left_batches):
                        action = batch_action[batch_ix].item()
                        log_prob = batched_log_prob[batch_ix]
                        actions[agent][batch_ix] = action
                        actual_batch_number = left_batches[batch_ix]
                        memory[agent][actual_batch_number].action = action
                        memory[agent][actual_batch_number].log_prob = log_prob
                        memory[agent][actual_batch_number].policy_distribution = final_policy_distribution[batch_ix]

                next_observations, rewards, dones = self.adv_piston2_batch_step(actions, step, time_penalty, early_reward_benefit)

                for agent in self.AGENT_NAMES:
                    for batch_ix in range(0, num_left_batches):
                        actual_batch_number = left_batches[batch_ix]
                        memory[agent][actual_batch_number].rewards = rewards[agent][batch_ix]
                    policies[agent].add_to_memory(memory[agent])

                for agent in self.AGENT_NAMES:
                    observations[agent] = next_observations[agent][~dones[agent]]

                self.update_adj_matrix()

            secondary_params = {}
            for agent in self.AGENT_NAMES:
                secondary_params[agent] = policies[agent].policy.state_dict().__str__()

            for agent in self.AGENT_NAMES:
                print(f"\t {agent} params unchanged: {initial_params[agent] == secondary_params[agent]}")

            for agent in self.AGENT_NAMES:
                optimizers[agent].zero_grad(set_to_none=False)
                policies[agent].set_batched_storage(self.BATCH_SIZE)

            epoch_data = {}
            iterations = 0
            for agent in self.AGENT_NAMES:
                loss, batched_mean_reward, batched_length_iteration = policies[agent].compute_loss()
                mean_reward = batched_mean_reward.mean()
                mean_iteration_length = batched_length_iteration.mean()
                iterations += mean_iteration_length
                epoch_data[agent] = [mean_reward, mean_iteration_length, batched_mean_reward, batched_length_iteration]
                #print('Performing backprop on %s' % (agent))
                if verbose:
                    print('\t Reward for %s: %f' % (agent, mean_reward))
                if agent == 'piston_2':
                    loss.backward(retain_graph=True)
            summary_stats.append(epoch_data)
            if verbose:
                print('\t *Team Mean Iterations: %s' % (iterations/self.N_AGENTS))

            for agent in self.AGENT_NAMES:
                torch.nn.utils.clip_grad_norm_(policies[agent].parameters(), max_grad_norm)

            if consensus_update:
                vnet_copies = {agent: policies[agent].policy.v_net.state_dict() for agent in policies.keys()}
                for agent in policies.keys():
                    neighbors_vnet = []
                    agent_num = int(agent.split('_')[1])
                    for i in [-1, 1]:
                        if 0 <= agent_num + i < self.N_AGENTS:
                            neighbors_vnet.append(vnet_copies['piston_%d' % (agent_num + i)])
                    policies[agent].consensus_update(neighbors_vnet)

            optimizers['piston_2'].step()
            if schedulers is not None:
                schedulers['piston_2'].step()

            for agent in self.AGENT_NAMES:
                policies[agent].clear_memory()

            # if epoch > 0:
            #     for agent in self.AGENT_NAMES:
            #         print(f"\t {agent} params unchanged in epoch: {last_epochs_params[agent] == secondary_params[agent]}")
            # last_epochs_params = secondary_params

        return policies, optimizers, summary_stats


if __name__ == '__main__':
    if torch.cuda.is_available():
        device=torch.device("cuda:0")
        print('**Using nvidia tesla')
    else:
        device=torch.device("cpu")
        print('**Using: cpu')
    n_agents = 5
    max_cycles = 200
    encoding_size = 300
    policy_latent_size = 20
    action_space = 3
    lr = 0.001
    epochs = 1000
    batch = 2

    hyper_params = {
        'encoding_size': 300,
        'policy_latent_size': 20,
        'lr': 0.001,
        'epochs': epochs,
        'n_agents': 5,
        'max_cycles': 200,
        'max_grad_norm': 0.75,
        'communicate': True,
        'transfer_experiment': {
            # 'name': '2023-03-14 19_01_25infopg',
            # 'name': '2023-03-20 18_24_09infopg',
            # 'name': './experiments/case_test/2023-03-20 18_24_09infopg',
            'name': '../experiments/final_models/pistonball/5agent_infopg(k=1)',
            'order': [0, 1, 2, 3, 4]
        },
        'time_penalty': 0.007,
        'early_reward_benefit': 0.25,
        'batch_size': batch,
        'k-levels': 1,
        'scheduler': None,
        'adv': 'normal',
        'seed': 1
    }

    env_params = {
        'n_pistons': n_agents, 'local_ratio': 1.0, 'time_penalty': 7e-3, 'continuous': False,
        'random_drop': True, 'random_rotate': True, 'ball_mass': 0.75, 'ball_friction': 0.3,
        'ball_elasticity': 1.5, 'max_cycles': max_cycles
    }

    user_params = {
        'device': device,
        'epochs': epochs,
        'verbose': True,
        'communicate': True,
        'max_grad_norm': 0.5,
        'time_penalty': 7e-3,
        'early_reward_benefit': 0.25,
        'consensus_update': False,
        'k-levels': 1
    }

    if 'seed' in hyper_params.keys():
        env = Piston5AgentByz(batch, env_params, seed=hyper_params['seed'])
    else:
        env = Piston5AgentByz(batch, env_params)

    if hyper_params['transfer_experiment'] is not None:
        policies, optimizers = create_policies_from_experiment(hyper_params, device)
    else:
        policies = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type="normal") for agent in env.get_agent_names()}
        optimizers = {agent: optim.Adam(policies[agent].parameters(), lr) for agent in env.get_agent_names()}
    policies['piston_2'] = PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type="normal")
    optimizers['piston_2'] = optim.Adam(policies['piston_2'].parameters(), lr)
    schedulers = {agent: optim.lr_scheduler.MultiStepLR(optimizer=optimizers[agent], milestones=[125, 600], gamma=0.99) for agent in env.get_agent_names()}
    policies, optimizers, summary_stats = env.loop(user_params, policies, optimizers, None)

    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    path = os.path.join('experiments', 'case_test', experiment_name + 'infopg')
    os.makedirs(path)

    with open('%s.pkl' % (os.path.join(path, 'data')), 'wb') as f:
        pickle.dump(summary_stats, f)

    with open('%s.txt' % (os.path.join(path, 'hyper_params')), 'w') as f:
        f.write(json.dumps(hyper_params))

    for agent in policies.keys():
        torch.save({
            'policy': policies[agent].state_dict(),
            'optimizer': optimizers[agent].state_dict()
        }, os.path.join(path, agent+'.pt'))