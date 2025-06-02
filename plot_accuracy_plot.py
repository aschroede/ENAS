# File: proxy_correlation_plot.py

import matplotlib.pyplot as plt
import numpy as np
from nasbench201_loader import load_nasbench201, query_nasbench201
from zero_cost_proxies import compute_non_gradient_proxies


def sample_architectures(api, arch_list, n=200):
    sampled = np.random.choice(arch_list, size=n, replace=False)
    return sampled


def evaluate_proxies_and_accuracy(api, archs, dataset='cifar10', hp='200'):
    final_epoch = {'12': 11, '36': 35, '90': 89, '200': 199}[hp]
    results = []
    for arch in archs:
        proxy_scores = compute_non_gradient_proxies(arch)

        train_acc, test_acc = query_nasbench201(api, arch)
        results.append((proxy_scores, test_acc))

    return results


def plot_proxy_vs_accuracy(results):
    proxy_names = list(results[0][0].keys())
    for proxy in proxy_names:
        scores = [r[0][proxy] for r in results]
        accs = [r[1] for r in results]
        plt.figure()
        plt.scatter(scores, accs, alpha=0.6)
        plt.xlabel(proxy)
        plt.ylabel("Validation Accuracy")
        plt.title(f"Correlation: {proxy} vs Accuracy")
        plt.grid(True)
        plt.savefig(f"plots/{proxy}_vs_accuracy.png")
        plt.close()


if __name__ == "__main__":
    api, arch_list = load_nasbench201()
    sampled_archs = sample_architectures(api, arch_list, n=200)
    results = evaluate_proxies_and_accuracy(api, sampled_archs)
    plot_proxy_vs_accuracy(results)
