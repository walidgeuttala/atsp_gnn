def find_lowest_val_loss(file_path):
    lowest_val_loss = float('inf')
    lowest_epoch = None

    with open(file_path, 'r') as f:
        saved_line = None
        for idx, line in enumerate(f):
            if idx < 27:
                idx += 1
                continue
            if "val_loss" in line:
                parts = line.split(' ')
                epoch = float(parts[1])
                val_loss = float(parts[5])
                
                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    lowest_epoch = epoch
                    saved_line = line

    return line

# Example usage:
# file_path = "../atsp_model_train_result/Oct15_14-45-35_HetroGATConcat_trained_ATSP50/trial_0/train_logs.txt"
file_path = "../atsp_model_train_result/Oct24_00-41-10_EdgePropertyPredictionModel3_trained_ATSP50/trial_0/train_logs.txt"
line = find_lowest_val_loss(file_path)
print(line)
