# File: main_cpu.py (Non-gradient-based search using NASBench-201)

import random
import json
from nasbench201_loader import load_nasbench201, query_nasbench201, mutate_architecture
from zero_cost_proxies import flops_proxy
from tqdm import tqdm

POPULATION_SIZE = 50
NUM_ITERATIONS = 100
TOURNAMENT_SIZE = 10

# NASBench-201 Documentation: https://github.com/D-X-Y/AutoDL-Projects/blob/main/docs/NAS-Bench-201.md
# Load NASBench-201
api, arch_list = load_nasbench201()

# Initialize population with random architectures
population = random.sample(arch_list, POPULATION_SIZE)
scores = [flops_proxy(arch) for arch in population]

for i in tqdm(range(NUM_ITERATIONS)):
    sample_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    parent_idx = max(sample_indices, key=lambda idx: scores[idx])
    parent = population[parent_idx]

    # Mutate and evaluate child
    child = mutate_architecture(parent)
    child_score = flops_proxy(child)

    # Add to population and remove oldest
    population.append(child)
    scores.append(child_score)
    population.pop(0)
    scores.pop(0)

# Evaluate best found architecture
best_idx = scores.index(max(scores))
best_arch = population[best_idx]

train_acc, test_acc = query_nasbench201(api, best_arch)
print("Best Architecture:", best_arch)
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Save results
with open("best_arch_cpu.json", "w") as f:
    json.dump({"arch": best_arch, "train_acc": train_acc, "test_acc": test_acc}, f)
