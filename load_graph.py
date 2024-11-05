import dgl
import torch
from utils import print_gpu_memory
from gnngls import datasets
from utils import *
from args import *
import json 

args_test = parse_args_test()
args = parse_args()
params = json.load(open(f'{args_test.model_path}/params.json'))
args = load_params(args, params)
# g = dgl.load_graphs("../tsp_input/graph_1000_half_st.dgl")[0][0]  # Load the first graph in the file
# print_gpu_memory('loadded')
print_gpu_memory('before loading the dataset')
test_set = datasets.TSPDataset(f'{args_test.data_path}/test.txt', args)
print_gpu_memory('after')


