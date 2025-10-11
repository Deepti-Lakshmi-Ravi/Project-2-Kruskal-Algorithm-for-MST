# Project 2 – Kruskal’s Algorithm for Minimum Spanning Tree

## Overview

This project analyzes the **time complexity of Kruskal’s Algorithm** for finding the **Minimum Spanning Tree (MST)** of a weighted connected graph.

Kruskal’s algorithm works by sorting all edges in non-decreasing order of weight and then adding edges one by one to the spanning tree, ensuring that no cycles are formed.  
A **Disjoint Set (Union–Find)** data structure is used to efficiently detect cycles.

The goal is to compare the **theoretical asymptotic runtime** `O(m log n)` with **experimental execution times**, validate the hypothesis, and visualize both results.

---

## Theoretical Basis

For a graph with:
- `n` = number of vertices  
- `m` = number of edges

Kruskal’s algorithm performs the following major operations:

1. **Sorting all edges:** `O(m log m)` → since `m ≤ n²`, this simplifies to `O(m log n)`
2. **Find and Union operations:** each taking near `O(log n)` on average  
3. **Total Time Complexity** is **T = O(m log(n) + n log(n))**.

Since for large graphs, m > n, the sorting step dominates. 
Thus, the time complexity can be simplified as:
                                   **T= O(m log(m))**


---

## Requirements

* Python 3.x  
* `matplotlib` library for plotting

Install the required library using:

```bash
pip install matplotlib
```

## Running the Code

1. Open a terminal and navigate to the project folder:

```bash
cd AsymptoticAnalysis
```

2. Run the Python script:

```bash
python analysis.py
```

## Input Sizes Tested

The following `n` values are used in the experiment:

```python
 n_values = [100, 200, 300, 400, 500, 800, 1000, 2000, 3000, 4000, 5000]
```
