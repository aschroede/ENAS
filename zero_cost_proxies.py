# File: zero_cost_proxies.py

def flops_proxy(arch_str):
    # Simple heuristic: favor more complex operations
    op_weights = {
        'none': 0,
        'skip_connect': 1,
        'avg_pool_3x3': 2,
        'nor_conv_1x1': 3,
        'nor_conv_3x3': 4
    }
    total = 0
    for op in arch_str.split('+'):
        for part in op.split('|'):
            if '~' in part:
                op_name = part.split('~')[0]
                total += op_weights.get(op_name, 0)
    return total


def compute_non_gradient_proxies(arch_str):
    # Example toy proxies
    return {
        "num_ops": arch_str.count("conv"),
        "num_edges": arch_str.count("~"),
        "num_nodes": arch_str.count('+') + 1,
    }