import random
import time
import math
import matplotlib.pyplot as plt
import unittest

 # Union-Find with path compression and union-by-rank
class DisjointSet:                      
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):                   # Find root with path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):               # Union two sets by rank
        rx, ry = self.find(x), self.find(y)
        
        if rx == ry:
            return False                  # Already in same set
        
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        
        return True

 # Kruskal's algorithm for MST
def kruskal(n, edges):                  
    edges.sort(key=lambda x: x[2])       # Sort by weight O(m log n)
    ds = DisjointSet(n)
    mst_weight = 0
    for u, v, w in edges:
        if ds.union(u, v):                # Add edge if no cycle           
            mst_weight += w
    return mst_weight

def generate_graph(n):                    # Generate complete graph with n vertices
    return [(i, j, random.randint(1, 100)) for i in range(n) for j in range(i+1, n)]

def experimental_runtime(n):              # Measure runtime in nanoseconds
    edges = generate_graph(n)
    start = time.perf_counter()
    kruskal(n, edges)
    elapsed_ns = (time.perf_counter() - start) * 1e9
    return elapsed_ns, len(edges)


def theoretical_operations(n):           # Calculate theoretical operations: m * log(n) where m = n(n-1)/2
    m = n * (n - 1) // 2
    return m * math.log2(n)

# UNIT TESTS
class TestDisjointSet(unittest.TestCase):  
    
    def test_find_single_element(self):
        ds = DisjointSet(5)
        self.assertEqual(ds.find(0), 0)
        self.assertEqual(ds.find(3), 3)
    
    def test_union_different_sets(self):
        ds = DisjointSet(5)
        result = ds.union(0, 1)
        self.assertTrue(result)
    
    def test_union_same_set(self):
        ds = DisjointSet(5)
        ds.union(0, 1)
        result = ds.union(0, 1)
        self.assertFalse(result)
    
    def test_path_compression(self):
        ds = DisjointSet(5)
        ds.union(0, 1)
        ds.union(1, 2)
        ds.find(0)
        self.assertEqual(ds.find(0), ds.find(2))
    
    def test_union_by_rank(self):
        ds = DisjointSet(10)
        ds.union(0, 1)
        ds.union(0, 2)
        ds.union(0, 3)
        self.assertEqual(ds.find(3), ds.find(0))
    
    def test_multiple_disjoint_sets(self):
        ds = DisjointSet(6)
        ds.union(0, 1)
        ds.union(2, 3)
        ds.union(4, 5)
        
        self.assertEqual(ds.find(0), ds.find(1))
        self.assertEqual(ds.find(2), ds.find(3))
        self.assertEqual(ds.find(4), ds.find(5))
        self.assertNotEqual(ds.find(0), ds.find(2))
        self.assertNotEqual(ds.find(2), ds.find(4))

class TestKruskal(unittest.TestCase):
    
    def test_single_vertex(self):
        edges = []
        result = kruskal(1, edges)
        self.assertEqual(result, 0)
    
    def test_two_vertices_one_edge(self):
        edges = [(0, 1, 5)]
        result = kruskal(2, edges)
        self.assertEqual(result, 5)
    
    def test_two_vertices_two_edges(self):
        edges = [(0, 1, 5), (0, 1, 3)]
        result = kruskal(2, edges)
        self.assertEqual(result, 3)
    
    def test_three_vertices_triangle(self):
        edges = [(0, 1, 1), (1, 2, 2), (0, 2, 3)]
        result = kruskal(3, edges)
        self.assertEqual(result, 3)
    
    def test_four_vertices_complete_graph(self):
        edges = [
            (0, 1, 1), (0, 2, 2), (0, 3, 3),
            (1, 2, 4), (1, 3, 5), (2, 3, 6)
        ]
        result = kruskal(4, edges)
        self.assertEqual(result, 6)
    
    def test_disconnected_components(self):
        edges = [(0, 1, 1), (2, 3, 2)]
        result = kruskal(4, edges)
        self.assertEqual(result, 3)
    
    def test_duplicate_weights(self):
        edges = [(0, 1, 5), (1, 2, 5), (0, 2, 5)]
        result = kruskal(3, edges)
        self.assertEqual(result, 10)
    
    def test_mst_multiple_edges(self):
        edges = [(0, 1, 1), (1, 2, 2), (0, 2, 3), (2, 3, 4)]
        result = kruskal(4, edges)
        self.assertEqual(result, 7)


class TestGraphGeneration(unittest.TestCase):
    
    def test_complete_graph_edge_count(self):
        for n in [2, 3, 4, 5, 10]:
            edges = generate_graph(n)
            expected_edges = n * (n - 1) // 2
            self.assertEqual(len(edges), expected_edges)
    
    def test_complete_graph_no_duplicates(self):
        edges = generate_graph(5)
        edge_set = {tuple(sorted([u, v])) for u, v, w in edges}
        self.assertEqual(len(edge_set), len(edges))
    
    def test_complete_graph_all_pairs(self):
        n = 4
        edges = generate_graph(n)
        edge_pairs = {tuple(sorted([u, v])) for u, v, w in edges}
        expected_pairs = {tuple(sorted([i, j])) for i in range(n) 
                         for j in range(i+1, n)}
        self.assertEqual(edge_pairs, expected_pairs)


# MAIN CODE

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':       # Run tests
        unittest.main(argv=[''], exit=False)
    else:                                                 # Run experimental analysis
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
        
        # Calculate scaling constant C
        C = sum(e*t for e, t in zip(exp_times_ns, theo_values)) / sum(t*t for t in theo_values)
        theo_scaled = [C*t for t in theo_values]
        
        # Print table
        for n, m, e_ns, t_ops, s_ns in zip(n_values, edge_counts, exp_times_ns, theo_values, theo_scaled):
            print(f"{n:>8} {m:>12} {e_ns:>20.0f} {t_ops:>25.0f} {s_ns:>25.0f}")
        
        print("="*100)
        print(f"\nScaling constant C = {C:.4f}")
        
        # Plot graph
        plt.figure(figsize=(10, 7))
        plt.plot(n_values, exp_times_ns, 'ro-', linewidth=3, markersize=8, 
                 label='Experimental Runtime (ns)', markeredgecolor='darkred')
        plt.plot(n_values, theo_scaled, 'b^-', linewidth=3, markersize=8, 
                 label='Scaled Theoretical Runtime', markeredgecolor='darkblue')
        
        plt.xlabel('Number of Vertices (n)', fontsize=14, fontweight='bold')
        plt.ylabel('Runtime (nanoseconds)', fontsize=14, fontweight='bold')
        plt.title("Kruskal's Algorithm: Experimental vs Theoretical Runtime", 
                  fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.4)
        plt.xticks(n_values, rotation=45)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.show()
