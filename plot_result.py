import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Sample data based on the table
atsp_sizes = [100, 150, 250, 500, 1000]
hetero_time_initial = [2.33, 4.45, 14.95, 82.46, None]
hetero_time_final = [6.68, 6.25, 15.05, 82.88, None]

matnet_time = [3.24, 4.24, 8.48, None, None]   # OOM for size 500 and 1000
glop_time = [14.25, 13.99, 14.86, 17.06, 22.57] 

matnet_gap_final = [1.84, 82.92, 181.68, None, None] # OOM for size 500 and 1000
glop_gap_final = [8.85, 17.91, 27.96, 38.96, 47.77] 
avg_gap_gnn = [11.49, 13.09, 13.04, 15.65, None]
avg_gap_final = [2.20, 4.75, 7.82, 9.38, None]

# Create the figure and two subplots
plt.figure(figsize=(14, 6))

# Plot execution time
plt.subplot(1, 2, 1)
plt.plot(atsp_sizes, hetero_time_initial, marker='o', label='Hetero_initial', color='blue')
plt.plot(atsp_sizes, hetero_time_final, marker='o', label='Hetero_final', color='yellow')
plt.plot(atsp_sizes, matnet_time, marker='o', label='MatNet', color='orange')
plt.plot(atsp_sizes, glop_time, marker='o', label='GLOP', color='green')
plt.title('Execution Time Comparison for ATSP', fontsize=16)
plt.xlabel('ATSP Size', fontsize=14)
plt.ylabel('Time (s)', fontsize=14)
plt.xticks(atsp_sizes)  # Set x-axis ticks at equal intervals based on atsp_sizes
plt.yscale('log', base=2)  # Set y-axis to log2 scale
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.gca().yaxis.set_major_formatter(ScalarFormatter())  # Format y-axis in normal numeric format

# Plot final average gap
plt.subplot(1, 2, 2)
plt.plot(atsp_sizes, avg_gap_gnn, marker='o', label='Hetero_initial', color='blue')
plt.plot(atsp_sizes, avg_gap_final, marker='o', label='Hetero_final', color='yellow')
plt.plot(atsp_sizes, matnet_gap_final, marker='o', label='MatNet', color='orange')
plt.plot(atsp_sizes, glop_gap_final, marker='o', label='GLOP', color='green')
plt.title('Final Average Gap Comparison for ATSP', fontsize=16)
plt.xlabel('ATSP Size', fontsize=14)
plt.ylabel('Final Avg Gap (%)', fontsize=14)
plt.xticks(atsp_sizes)  # Set x-axis ticks at equal intervals based on atsp_sizes
plt.yscale('log', base=2)  # Set y-axis to log2 scale
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.gca().yaxis.set_major_formatter(ScalarFormatter())  # Format y-axis in normal numeric format

# Adjust layout and save to PDF
plt.tight_layout()
plt.savefig('result_plot.pdf')  # Save as PDF


plt.show()  # Display the plot
plt.close()  # Close the figure to free memory
