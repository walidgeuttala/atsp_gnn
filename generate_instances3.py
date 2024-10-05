#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import multiprocessing as mp
import pathlib
import uuid

import networkx as nx
import numpy as np

import gnngls
from gnngls import datasets


def parse_graph_from_content(content):
    lines = content.splitlines()
    edge_weight_section_start = False
    dimension = 0
    weights = []

    for line in lines:
        if line.startswith("DIMENSION"):
            dimension = int(line.split()[-1])
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            edge_weight_section_start = True
        elif edge_weight_section_start:
            if line.strip() == "EOF":
                break
            weights.extend(map(float, line.split()))

    G = nx.DiGraph()
    G.add_nodes_from(range(dimension))
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                G.add_edge(i, j, weight=weights[i * dimension + j])
    
    return G


def prepare_instance(G):
    datasets.set_features(G)
    datasets.set_labels(G)
    return G


def get_graphs_from_files(folder_path):
    for file_path in pathlib.Path(folder_path).glob("*.atsp"):
        with open(file_path, 'r') as f:
            content = f.read()
            G = parse_graph_from_content(content)

            in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
            nx.set_edge_attributes(G, in_solution, 'in_solution')


            yield G


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset from .atsp files in a folder.')
    parser.add_argument('n_samples', type=int, help='Number of samples to generate')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing .atsp files')
    parser.add_argument('output_dir', type=pathlib.Path, help='Output directory for the generated dataset')
    args = parser.parse_args()

    if args.output_dir.exists():
        raise Exception(f'Output directory {args.output_dir} exists.')
    else:
        args.output_dir.mkdir()

    pool = mp.Pool(processes=None)
    graphs = get_graphs_from_files(args.folder_path)

    # Limiting the number of samples processed
    for idx, G in enumerate(pool.imap_unordered(prepare_instance, graphs)):
        if idx >= args.n_samples:
            break
        nx.write_gpickle(G, args.output_dir / f'{uuid.uuid4().hex}.pkl')

    pool.close()
    pool.join()
