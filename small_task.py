import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from gnngls import  datasets
import pickle
# Define the folder path
folder_path = '../tsp_lib_test_with_regret2/'
plots_folder_path = os.path.join("", 'plots2_degree')
value = 'weight'
# Create the 'plots' directory if it doesn't exist
os.makedirs(plots_folder_path, exist_ok=True)
cnt = 0
scalers = pickle.load(open("../tsp_lib_test_with_regret2/scalers.pkl", 'rb'))
# Iterate through each file in the directory
test_set = datasets.TSPDataset(folder_path+'train.txt')
for filename in os.listdir(folder_path):
    cnt += 1
    if filename.endswith('.pkl'):
        # Construct the full file path
        if filename == "scalers.pkl":
            continue
        file_path = os.path.join(folder_path, filename)
        
        # Load the graph using networkx
        graph = nx.read_gpickle(file_path)
        H = test_set.get_test_scaled_features_not_samesize_graphs(graph)
        # Get the number of nodes in the graph
        num_nodes = graph.number_of_nodes()
        
        # Collect the 'regret' feature from each node
        regret, _ = nx.attr_matrix(graph, value)
        regret = scalers[value].inverse_transform(np.asarray(regret))
        upper_tri = np.triu(regret, k=1)

        # Extract the lower triangular part without the diagonal
        lower_tri = np.tril(regret, k=-1)

        # Combine the upper and lower triangular parts
        combined = upper_tri + lower_tri

        # Flatten the combined array and remove zeros
        regret = combined[combined != 0].tolist()
        #regret = regret.flatten().tolist()

        # Plot the distribution of the 'regret' feature
        plt.figure()
        plt.hist(regret, bins=20, edgecolor='black')
        plt.title(f'Regret Distribution for {num_nodes} Nodes')
        plt.xlabel('Regret')
        plt.ylabel('Frequency')
        
        # Save the plot to a file in the 'plots' directory
        plot_filename = f'regret_distribution_{num_nodes}_{cnt}.png'
        plot_file_path = os.path.join(plots_folder_path, plot_filename)
        plt.savefig(plot_file_path)
        plt.close()

        # SEcond plot
        plt.figure()
        plt.hist(regret, bins=20, edgecolor='black')
        plt.title(f'Regret before Distribution for {num_nodes} Nodes')
        plt.xlabel('Regret')
        plt.ylabel('Frequency')
        
        # Save the plot to a file in the 'plots' directory
        plot_filename = f'regret_distribution_{num_nodes}_{cnt}_before.png'
        plot_file_path = os.path.join(plots_folder_path, plot_filename)
        plt.savefig(plot_file_path)
        plt.close()
        
        
        print(f'Saved plot as {plot_filename}')



