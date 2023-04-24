import pickle as pkl
import statistics

folder_name = "./experiments/case_test/"
# filename = "2023-04-18 00_08_51infopg/data.pkl"
# filename = "2023-04-18 01_41_23infopg/data.pkl"
# filename = "2023-04-18 18_42_49infopg_eval/data.pkl"
# filename = "2023-04-19 11_07_01infopg_eval/data.pkl"
# filename = "2023-04-20 16_34_15infopg_eval/data.pkl"
filename = "2023-04-20 17_17_05infopg_eval/data.pkl"

with open(folder_name + filename, "rb") as f:
    data = pkl.load(f)

piston_2_rewards = []
piston_2_episode_lengths = []

for epoch in data:
    piston_2_data = epoch["piston_2"]
    piston_2_rewards.extend(piston_2_data[2])
    piston_2_episode_lengths.extend(piston_2_data[3])

piston_2_rewards = sorted(piston_2_rewards)
piston_2_episode_lengths = sorted(piston_2_episode_lengths)

print(f"\tMean iteration length {statistics.mean(piston_2_episode_lengths)}")
print(f"\tMedian iteration length {statistics.median(piston_2_episode_lengths)}")
print(f"\tMean piston 2 reward {statistics.mean(piston_2_rewards)}")
print(f"\tMedian piston 2 reward {statistics.median(piston_2_rewards)}")


