import random
import time
import math
import matplotlib.pyplot as plt

class DisjointSet:                     #Disjoint Set(Union-Find)
    def __init__(self, n):
        self.parent = list(range(n))   # Parent array 
        self.rank = [0] * n            # Rank array 
    
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

def kruskal(n, edges):              # Kruskal's Algorithm for MST
    edges.sort(key=lambda x: x[2])  # Sort by weight - O(m log m)
    ds = DisjointSet(n)
    mst_weight = 0
    for u, v, w in edges:
        if ds.union(u, v):
            mst_weight += w
    return mst_weight

def generate_graph(n):              #Generate a complete graph with n vertices
    return [(i, j, random.randint(1, 100)) for i in range(n) for j in range(i+1, n)]

def experimental_runtime(n):        #Measure experimental runtime
    edges = generate_graph(n)
    start = time.perf_counter()
    kruskal(n, edges)
    elapsed_ns = (time.perf_counter() - start) * 1e9
    return elapsed_ns, len(edges)

def theoretical_operations(n):
    m = n * (n - 1) // 2
    return m * math.log2(n)  # Using m log n to match textbook

# Test values
n_values = [100, 200, 300, 400, 500, 800, 1000, 2000, 3000, 4000, 5000]
exp_times_ns = []
theo_values = []
edge_counts = []

print("="*100)
print(f"{'n':>8} {'m (edges)':>12} {'Experimental Runtime(ns)':>20} {'Theoretical Values':>25} {'Scaled Theoretical Values(ns)':>25}")
print("="*100)

# Collect data
for n in n_values:
    exp_ns, m = experimental_runtime(n)
    theo = theoretical_operations(n)
    exp_times_ns.append(exp_ns)
    theo_values.append(theo)
    edge_counts.append(m)

# Scaling
C = sum(e*t for e, t in zip(exp_times_ns, theo_values)) / sum(t*t for t in theo_values)
theo_scaled = [C*t for t in theo_values]

# Print table with edge counts
for n, m, e_ns, t_ops, s_ns in zip(n_values, edge_counts, exp_times_ns, theo_values, theo_scaled):
    print(f"{n:>8} {m:>12} {e_ns:>20.0f} {t_ops:>25.0f} {s_ns:>25.0f}")

print("="*100)
print(f"\nScaling constant C = {C:.4f}")

# Plotting the graph
plt.figure(figsize=(12, 7))
plt.plot(n_values, exp_times_ns, 'ro-', linewidth=3, markersize=8, 
         label='Experimental Runtime (ns)', markeredgecolor='darkred')
plt.plot(n_values, theo_scaled, 'b^-', linewidth=3, markersize=8, 
         label='Scaled Theoretical Runtime', markeredgecolor='darkblue')

plt.xlabel('Number of Vertices (n)', fontsize=14, fontweight='bold')
plt.ylabel('Runtime (nanoseconds)', fontsize=14, fontweight='bold')
plt.title("Kruskal's Algorithm: Experimental vs Theoretical Runtime)", 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.4)
plt.xticks(n_values, rotation=45)
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.show()
