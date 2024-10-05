import os

def read_and_modify_files(input_folder, output_folder, n):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files with '.txt' extension in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # Read the content of the input file
            with open(input_file_path, "r") as file:
                lines = file.readlines()

            # Find where to insert the 'regret' section
            with open(output_file_path, "w") as new_file:
                for line in lines:
                    
                    # Look for the line with "regret_pred:"
                    if line.startswith("regret_pred:"):
                        # After this line, we add the regret section with zeros
                        new_file.write("regret:\n")
                        for _ in range(n):
                            new_file.write(" ".join([f"{0.00000000:.8f}" for _ in range(n)]) + "\n")
                        new_file.write("\n")
                    new_file.write(line)
               
            print(f"Processed file: {file_name} and saved to {output_folder}")

# Input parameters
input_folder = "../output_ATSP_samples_1000_size_100"  # Replace with actual input folder path
output_folder = "../output_ATSP_samples_1000_size_100_adding_regret"  # Replace with actual output folder path
n = 100 # Replace with the desired number of rows and columns for the regret matrix

# Run the function
read_and_modify_files(input_folder, output_folder, n)

