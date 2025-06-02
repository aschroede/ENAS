import random
from nasbench201_loader import mutate_architecture, query_nasbench201, load_nasbench201
import matplotlib.pyplot as plt
import itertools

def evolutionary_search(api, arch_list, pop_size, mutation_rate, generations):
    population = random.sample(arch_list, pop_size)
    scores = []

    # Evaluate initial population
    population_scores = [(arch, query_nasbench201(api, arch)[0]) for arch in population]

    for gen in range(generations):
        population_scores.sort(key=lambda x: x[1], reverse=True)
        parents = population_scores[:int(pop_size / 2)]

        offspring = []
        while len(offspring) < pop_size - len(parents):
            parent_arch = random.choice(parents)[0]
            if random.random() < mutation_rate:
                child_arch = mutate_architecture(parent_arch)
                offspring.append((child_arch, query_nasbench201(api, child_arch)[0]))

        population_scores = parents + offspring
        best_score = max(population_scores, key=lambda x: x[1])[1]
        scores.append(best_score)

    return scores[-1]  # Return best after all generations



api, arch_list = load_nasbench201()

pop_sizes = [10, 20, 30]
mutation_rates = [0.2, 0.5, 0.8]
generations = [5, 10, 20]

results = []

for pop_size, mut_rate, gen in itertools.product(pop_sizes, mutation_rates, generations):
    print(f"Running: pop={pop_size}, mut={mut_rate}, gen={gen}")
    best_acc = evolutionary_search(api, arch_list, pop_size, mut_rate, gen)
    results.append((pop_size, mut_rate, gen, best_acc))

# Plot
fig, ax = plt.subplots()
xs = [f"P{p}-M{m}-G{g}" for p, m, g, _ in results]
ys = [acc for _, _, _, acc in results]
ax.bar(xs, ys)
ax.set_ylabel("Best Validation Accuracy")
ax.set_xlabel("Config (Population-Mutation-Generations)")
ax.set_title("Evolutionary Search Sweep Results")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/evolution_sweep.png")
