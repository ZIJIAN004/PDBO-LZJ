import gzip
import csv
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from collections import defaultdict




def parse_gset(ins_id):
    nx_graph = nx.Graph()
    with open(f'./instance/Gset/G{ins_id}.txt', 'r') as f:
        lines = f.readlines()
    n, m = map(int, lines[0].split())
    nx_graph.add_nodes_from(range(n))
    assert len(lines) == (m + 1)
    for line in lines[1:]:
        u, v, w = map(int, line.split())

        nx_graph.add_edge(u - 1, v - 1, weight=w)
    return nx_graph




def random_graph(n, d, seed=0):
    nx_temp = nx.random_regular_graph(d=d, n=n, seed=seed)
    sorted_nodes = sorted(nx_temp.nodes())
    mapping = {old: new for new, old in enumerate(sorted_nodes)}
    G_temp = nx.relabel_nodes(nx_temp, mapping)
    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(G_temp.nodes()))
    nx_graph.add_edges_from(G_temp.edges())
    for u, v in nx_graph.edges():
        nx_graph[u][v]['weight'] = 1
    return nx_graph



def get_matrix(nx_G):
    n_nodes = len(nx_G.nodes())
    W_mat = np.zeros((n_nodes, n_nodes))
    for (u, v, val) in nx_G.edges(data = True):
        W_mat[u][v] = val['weight'] / 2
        W_mat[v][u] = val['weight'] / 2
    return W_mat


def postprocess(result, graph):
    """
    helper function to postprocess MIS results

    Input:
        best_bitstring: bitstring as torch tensor
    Output:
        size_mis: Size of MIS (int)
        ind_set: MIS (list of integers)
        number_violations: number of violations of ind.set condition
    """
    maxcut = 0
    for (u, v, val) in graph.edges(data=True):
        wt = val['weight']
        if result[u] != result[v]:
            maxcut = maxcut + wt
    return maxcut



def parses_COLOR(name):
    """Reads a COLOR graph from file."""
    nx_graph = nx.Graph()
    with open(f"./instance/COLOR/{name}.col") as f:
        for line in f:
            tokens = line.split()
            if tokens[0] == "p":
                n = int(tokens[2])
                nx_graph.add_nodes_from(range(n))
            elif tokens[0] == "e":
                w = 1

                nx_graph.add_edge(int(tokens[1]) - 1, int(tokens[2]) - 1, weight=w)
    mapping = {old: new for new, old in enumerate(sorted(nx_graph.nodes()), start=0)}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    return nx_graph


def generate_MIS(graph, penalty=2):
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    c = - np.full(n, 1, dtype=np.float32)
    indices, values = [], []
    for (u, v) in graph.edges():
        values.append(penalty)
        indices.append([u, v])
    value = np.array(values, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32).T  # 转置操作相当于 PyTorch 的 transpose(0, 1)

    Q_indices = np.concatenate([indices, np.flip(indices, axis=0)], axis=1)
    Q_values = np.concatenate([value, value]) / 2
    Q_sparse = coo_matrix((Q_values, Q_indices), shape=(n, n))

    data = {'indices': indices, 'value': value, 'bounds': (0, 1), 'num_nodes': n, 'num_edges': m, 'num_vars': n,
            'Q_indices': Q_indices, 'Q_values': Q_values, 'c': c, 'Q_sparse': Q_sparse}
    return data


def generate_Max_cut(graph):
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    Q_values, Q_indices = [], []
    for (i, j, w) in graph.edges(data=True):
        Q_values.append(w['weight'])
        Q_values.append(w['weight'])
        Q_indices.append([i, j])
        Q_indices.append([j, i])

    Q_values = np.array(Q_values, dtype=np.float32)
    Q_indices = np.array(Q_indices, dtype=np.int32).T

    Q_sparse = coo_matrix((Q_values, Q_indices), shape=(n, n))
    c = -np.array(Q_sparse.sum(axis=1), dtype=np.float32).flatten()
    data = {'num_nodes': n, 'num_edges': m, 'num_vars': n,
            'Q_indices': Q_indices, 'Q_values': Q_values, 'c': c, 'Q_sparse': Q_sparse}
    return data

def generate_max_sat(path):
    file = open(path, 'r')
    f = []

    for line in file:
        s = line.split()
        if not len(s) == 0 and not line[0] == 'c' and not s[0] == 'p':
            if s[-1] == '0' and len(s) > 1:
                clause = [int(l) for l in s[:-1]]
                f.append(clause)
    file.close()
    cnf = np.array([np.int64(c) for c in f])
    num_vars = np.max([np.abs(c).max() for c in cnf])
    data = {'num_vars': num_vars,'CNF': cnf}
    return data


def _labs_z_index(n, i, k):
    """0-based variable index for z[i, k], where i + k < n."""
    offset = n
    for prev_k in range(1, k):
        offset += n - prev_k
    return offset + i


def generate_LABS(n, penalty=10000):
    linear = defaultdict(float)
    quadratic = defaultdict(float)
    feasible_triples = []
    offset = 0.0

    def add_linear(var, coeff):
        linear[var] += coeff

    def add_quadratic(var1, var2, coeff):
        if var1 == var2:
            add_linear(var1, coeff)
            return
        key = tuple(sorted((var1, var2)))
        quadratic[key] += coeff

    for k in range(1, n):
        expr = defaultdict(float)
        const = n - k
        for i in range(n - k):
            z = _labs_z_index(n, i, k)
            feasible_triples.append([z, i, i + k])
            expr[z] += 4.0
            expr[i] -= 2.0
            expr[i + k] -= 2.0

            add_linear(z, 3.0 * penalty)
            add_quadratic(z, i, -2.0 * penalty)
            add_quadratic(z, i + k, -2.0 * penalty)
            add_quadratic(i, i + k, penalty)

        offset += const ** 2
        items = list(expr.items())
        for var, coeff in items:
            add_linear(var, 2.0 * const * coeff + coeff * coeff)
        for idx, (var1, coeff1) in enumerate(items):
            for var2, coeff2 in items[idx + 1:]:
                add_quadratic(var1, var2, 2.0 * coeff1 * coeff2)

    num_z = n * (n - 1) // 2
    num_vars = n + num_z
    c = np.zeros(num_vars, dtype=np.float32)
    for var, coeff in linear.items():
        c[var] = coeff

    q_indices = []
    q_values = []
    for (var1, var2), coeff in quadratic.items():
        if coeff == 0:
            continue
        q_indices.append([var1, var2])
        q_values.append(coeff / 2.0)
        q_indices.append([var2, var1])
        q_values.append(coeff / 2.0)

    Q_indices = np.array(q_indices, dtype=np.int32).T
    Q_values = np.array(q_values, dtype=np.float32)
    Q_sparse = coo_matrix((Q_values, Q_indices), shape=(num_vars, num_vars))
    data = {
        'num_vars': num_vars,
        'num_x_vars': n,
        'num_z_vars': num_z,
        'Q_indices': Q_indices,
        'Q_values': Q_values,
        'c': c,
        'Q_sparse': Q_sparse,
        'objective_offset': float(offset),
        'penalty': penalty,
        'labs_feasible_triples': np.array(feasible_triples, dtype=np.int32),
    }
    return data


def evaluate_LABS_bits(bits):
    spins = 2 * np.array(bits, dtype=np.int32) - 1
    n = len(spins)
    energy = 0
    for k in range(1, n):
        autocorrelation = int(np.dot(spins[:n - k], spins[k:]))
        energy += autocorrelation ** 2
    return energy
