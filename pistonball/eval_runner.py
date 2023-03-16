def eval(experiment_dir, k_levels=1, adv='normal', use_critic=False, consenses=False):
    device=torch.device("cpu")
    print('**Using: cpu for inference')

    env = Piston5AgentCase(batch, env_params)
    policies = {agent: PistonPolicyCASE(device) for agent in env.get_agent_names()}

    print('Evaluating: ', experiment_dir)
    with open(os.path.join('..', 'experiments','final_models', 'pistonball', experiment_dir, 'combined_model.pt'), 'rb') as f:
        d = torch.load(f, map_location=device)
        model.load_state_dicts(d['policy'])

if __name__ == '__main__':
    n_agents = 5
    max_cycles = 200
    encoding_size = 300
    policy_latent_size = 20
    action_space = 3
    lr = 0.001
    epochs = 1000
    batch = 2

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