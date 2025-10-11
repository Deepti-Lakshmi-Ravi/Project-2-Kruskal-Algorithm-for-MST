import random
import time
import math
import matplotlib.pyplot as plt

class DisjointSet:                            #Disjoint Set for Union Find
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

def kruskal(n, edges):                         #Kruskal's Algorithm
    edges.sort(key=lambda x: x[2])
    ds = DisjointSet(n)
    mst_weight = 0
    for u, v, w in edges:
        if ds.union(u, v):
            mst_weight += w
    return mst_weight

def generate_graph(n):
    return [(i, j, random.randint(1, 100)) for i in range(n) for j in range(i+1, n)]

def experimental_runtime(n):
    edges = generate_graph(n)
    start = time.perf_counter()
    kruskal(n, edges)
    elapsed_ns = (time.perf_counter() - start) * 1e9
    return elapsed_ns, len(edges)

def theoretical_operations(n):
    m = n * (n - 1) // 2
    return m * math.log2(m)

n_values = [100, 200, 300, 400, 500, 800, 1000, 2000, 3000, 4000, 5000]
exp_times_ns = []
theo_values = []

print(f"{'n':>8} {'Experimental(ns)':>20} {'Theoretical Operations':>25} {'Scaled Values(ns)':>20}")
print("="*90)

for n in n_values:
    exp_ns, m = experimental_runtime(n)
    theo = theoretical_operations(n)
    exp_times_ns.append(exp_ns)
    theo_values.append(theo)

C = sum(e*t for e, t in zip(exp_times_ns, theo_values)) / sum(t*t for t in theo_values)      # Scaling the theoretical values
theo_scaled = [C*t for t in theo_values]

# Print table
for n, e_ns, t_ops, s_ns in zip(n_values, exp_times_ns, theo_values, theo_scaled):
    print(f"{n:8d} {int(e_ns):20d} {int(t_ops):25d} {int(s_ns):20d}")

print("="*90)
print(f"\nScaling constant C = {C:.4f}\n")

# Plotting the Graph
plt.figure(figsize=(10, 6))
plt.plot(n_values, exp_times_ns, 'ro-', linewidth=2, markersize=8, label='Experimental Runtime (ns)')
plt.plot(n_values, theo_scaled, 'b^-', linewidth=2, markersize=8, label='Scaled Theoretical Runtime')
plt.xlabel('Number of Vertices (n)', fontsize=12)
plt.ylabel('Runtime (nanoseconds)', fontsize=12)
plt.title("Kruskal's Algorithm: Experimental vs Scaled Theoretical Runtime", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.xticks(n_values, rotation=45)

plt.tight_layout()
plt.show()
