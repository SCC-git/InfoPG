import pickle as pkl
import statistics
import matplotlib.pyplot as plt

# folder_name = "./experiments/pistonball/"
folder_name = "../experiments/final_models/pistonball/5agent_infopg_nocritic(k=1)/"
# filename = "2023-03-23 04_32_26/" + "data.pkl"
filename = "data.pkl"

with open(folder_name + filename, "rb") as f:
    data = pkl.load(f)

cum_rewards = []
ep_lengths = []

for epoch in data:
    ep_rewards = []
    for piston in epoch:
        ep_rewards.append(epoch[piston][0])
    cum_rewards.append(sum(ep_rewards))

    ep_lengths.append(epoch['piston_0'][1]) #append mean ep length

print(f"Episode rewards: {cum_rewards}")
print(f"Episode lengths: {ep_lengths}")

plt.plot(cum_rewards)
plt.title('Cumulative rewards over training')
plt.xlabel('Epoch')
plt.ylabel('Cumulative team reward')
plt.show()

plt.plot(ep_lengths)
plt.title('Episode lengths over training')
plt.xlabel('Epoch')
plt.ylabel('Episode length')
plt.show()

