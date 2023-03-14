
def get_render_run(experiment_dir, k_levels=1):
    device=torch.device("cpu")
    print('**Using: cpu for inference')
    print('**Just getting a rendering!')
    model = Combined_Walker_Helper(device, k_levels)