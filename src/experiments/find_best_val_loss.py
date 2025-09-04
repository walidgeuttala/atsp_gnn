def find_lowest_val_loss(file_path):
    lowest_val_loss = float('inf')
    lowest_gap   = None
    saved_line = None
    with open(file_path, 'r') as f:
        
        for idx, line in enumerate(f):
            if idx < 27:
                idx += 1
                continue
            if "val_loss" in line:
                parts = line.split(' ')
                epoch = float(parts[1])
                val_loss = round(float(parts[5]), 4)
                avg_gap = round(float(parts[7]), 2)
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    saved_line = line
                    lowest_gap = avg_gap
                elif val_loss == lowest_val_loss and avg_gap < lowest_gap:
                    lowest_gap = avg_gap
                    saved_line = line
                    
    output = saved_line.split(' ')
    return output

# Example usage:
# file_path = "../atsp_model_train_result/Oct15_14-45-35_HetroGATConcat_trained_ATSP50/trial_0/train_logs.txt"
file_path = "../atsp_model_train_result/Oct24_20-58-12_HetroGATSum_trained_ATSP50/trial_0/train_logs.txt"
output = find_lowest_val_loss(file_path)
print(f'{float(output[5])} {round(float(output[7]), 2)} {round(float(output[13])*100, 2)} {round(float(output[-1])*100, 2)}')
