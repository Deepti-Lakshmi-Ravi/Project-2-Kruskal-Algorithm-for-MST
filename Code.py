import random
import time
import math
import matplotlib.pyplot as plt

# ============ DISJOINT SET (UNION-FIND) ============
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

# ============ KRUSKAL'S ALGORITHM ============
def kruskal(n, edges):
    
    if n < 0 or not isinstance(n, int):
        raise ValueError("n must be a non-negative integer")
    if not isinstance(edges, list):
        raise ValueError("edges must be a list")
    
    edges.sort(key=lambda x: x[2])
    ds = DisjointSet(n)
    mst_weight = 0
    
    for u, v, w in edges:
        if ds.union(u, v):
            mst_weight += w
    
    return mst_weight

# ============ GRAPH GENERATION ============
def generate_graph(n):
    
    return [(i, j, random.randint(1, 100)) for i in range(n) for j in range(i+1, n)]

# ============ EXPERIMENTAL RUNTIME ============
def experimental_runtime(n, trials=5):
    
    times = []
    
    for _ in range(trials):
        edges = generate_graph(n)
        start = time.perf_counter()
        kruskal(n, edges)
        elapsed_ns = (time.perf_counter() - start) * 1e9
        times.append(elapsed_ns)
    
    avg_time = sum(times) / len(times)
    std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return avg_time, len(edges), std_dev

# ============ THEORETICAL OPERATIONS ============
def theoretical_operations(n):
    m = n * (n - 1) // 2
    return m * math.log2(m)

# ============ MAIN EXPERIMENT ============
n_values = [100, 200, 300, 400, 500, 800, 1000, 2000, 3000, 4000, 5000]
exp_times_ns = []
theo_values = []
std_devs = []

# Table Header
print(f"{'n':>8} {'Experimental Runtime(ns)':>20} {'Theoretical Values':>20} {'Scaled Theoretical Values(ns)':>20}")
print("=" * 110)

# Collect data
for n in n_values:
    exp_ns, m, std_dev = experimental_runtime(n, trials=5)
    theo = theoretical_operations(n)
    exp_times_ns.append(exp_ns)
    theo_values.append(theo)
    std_devs.append(std_dev)

# ============ SCALING ============
C = sum(e * t for e, t in zip(exp_times_ns, theo_values)) / sum(t * t for t in theo_values)
theo_scaled = [C * t for t in theo_values]

# Print table
for n, e_ns, std_d, t_ops, s_ns in zip(n_values, exp_times_ns, std_devs, theo_values, theo_scaled):
    print(f"{n:8d} {int(e_ns):20d} {int(std_d):15d} {int(t_ops):20d} {int(s_ns):20d}")

print(f"\nScaling constant C = {C:.4f}\n")

# ============ PLOTTING ============
plt.figure(figsize=(12, 7))
plt.plot(n_values, exp_times_ns, 'ro-', linewidth=2, markersize=8, 
         label='Experimental Runtime (ns)')
plt.plot(n_values, theo_scaled, 'b^-', linewidth=2, markersize=8, 
         label='Scaled Theoretical Runtime')

# Add error bars for experimental data
plt.errorbar(n_values, exp_times_ns, yerr=std_devs, fmt='none', 
             ecolor='red', alpha=0.3, capsize=5)

plt.xlabel('Number of Vertices (n)', fontsize=12)
plt.ylabel('Runtime (nanoseconds)', fontsize=12)
plt.title("Kruskal's Algorithm: Experimental vs Scaled Theoretical Runtime", fontsize=14)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)

# X-axis: show all values
plt.xticks(n_values, rotation=45)

plt.tight_layout()
plt.show()
