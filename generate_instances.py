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

import linecache

def prepare_instance(G):
    datasets.set_features(G)
    datasets.set_labels2(G)
    return G

def get_solved_instances(n_nodes, n_instances):
    for _ in range(n_instances):
        G = nx.Graph()

        coords = np.random.random((n_nodes, 2))
        for n, p in enumerate(coords):
            G.add_node(n, pos=p)

        for i, j in itertools.combinations(G.nodes, 2):
            w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos'])
            G.add_edge(i, j, weight=w)

        opt_solution = gnngls.optimal_tour(G)
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')

        yield G


def get_solved_instances2(n_nodes, n_instances, all_instances):
    #all_instances = './tsplib95_10000_instances_64_node/all_instances_lower_triangle_tour_cost.txt'
    # Open the file in read mode
   
    for i in range(n_instances):
        line = linecache.getline(all_instances, i+2).strip()
        
        G = nx.DiGraph()
        adj, opt_solution, cost = line.split(',')
        adj = adj.split(' ')[:-1]
        print(len(adj))
        n_nodes = len(opt_solution.split()) - 1
        print(n_nodes)
        G.add_nodes_from(range(n_nodes))
        opt_solution = [int(x) for x in opt_solution.split()]
       
        # Add the edges for the DiGraph and be sure that does not have self loops in the node
        for j in range(n_nodes):
            for k in range(n_nodes):
                w = float(adj[j*n_nodes+k])
                if j != k:
                    G.add_edge(j, k, weight=w)
            
        in_solution = gnngls.tour_to_edge_attribute(G, opt_solution)
        nx.set_edge_attributes(G, in_solution, 'in_solution')
        opt_cost = gnngls.optimal_cost(G, weight='weight')
        if opt_cost != cost:
            print('does not match opt_cost:{opt_cost} cost:{cost}')
        yield G

import networkx as nx
import os
import json

# reading the new data format from json, jsut for testing
def get_solved_instances3(n_nodes, n_instances, all_instances):
    for i in range(n_instances):
        # Construct the file path
        file_path = os.path.join(all_instances, f'solved_instance_{i}.json')
        
        # Open and load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Parse fields from the JSON data
        adjacency_matrix = data.get("adjacency_matrix")
        tour = data.get("tour", [])
        cost = data.get("cost", None)
        
        # Validate the adjacency matrix and the number of nodes
        if not adjacency_matrix or len(adjacency_matrix) != len(adjacency_matrix[0]):
            print(f"Error in instance {i}: Invalid adjacency matrix.")
            continue
        
        dimension = len(adjacency_matrix)  # Number of nodes
        if n_nodes and dimension != n_nodes:
            print(f"Warning: Instance {i} has {dimension} nodes, expected {n_nodes} nodes.")
        
        # Create a directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(dimension))

        # Add the edges from the parsed adjacency matrix
        for j in range(dimension):
            for k in range(dimension):
                w = float(adjacency_matrix[j][k])
                if j != k:
                    G.add_edge(j, k, weight=w)

        # If no tour (optimal solution) is found, skip the graph
        if not tour:
            print(f"No tour found for instance {i}.")
            continue

        # Convert the tour to integers (if needed)
        tour = [int(x) for x in tour]

        # Mark edges in the solution
        in_solution = gnngls.tour_to_edge_attribute(G, tour)
        nx.set_edge_attributes(G, in_solution, 'in_solution')
        # Calculate and compare the optimal cost
        opt_cost = gnngls.optimal_cost(G, weight='weight')
        if round(opt_cost, 3) != round(cost, 3):
            print(f"Instance {i}: Cost mismatch! Computed: {opt_cost}, Expected: {cost}")

        yield G



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a dataset.')
    parser.add_argument('n_samples', type=int)
    parser.add_argument('n_nodes', type=int)
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=pathlib.Path)
    args = parser.parse_args()

    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = mp.Pool(processes=None)
    instance_gen = get_solved_instances3(args.n_nodes, args.n_samples, args.input_dir)
    # Process and save instances
    try:
        for G in pool.imap_unordered(prepare_instance, instance_gen):
            output_file = args.output_dir / f'{uuid.uuid4().hex}.pkl'
            nx.write_gpickle(G, output_file)
    except Exception as e:
        print(f"Error occurred during processing: {e}")
    finally:
        pool.close()
        pool.join()

