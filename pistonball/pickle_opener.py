import pickle as pkl

folder_name = "./experiments/pistonball/2023-03-14 12_22_31/"
filename = "hyper_params.pkl"

with open(folder_name + filename, "rb") as f:
    data = pkl.load(f)

print(data)