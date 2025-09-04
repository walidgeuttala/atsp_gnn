import matplotlib.pyplot as plt

# Sample data based on the table
atsp_sizes = [150, 250, 500, 1000]
hetero_time_initial = [4.45, 14.95, 82.46, None]
hetero_time_final = [6.25, 15.05, 82.88, None]

matnet_time = [4.24, 8.48, None, None]   # OOM for size 500 and 1000
glop_time = [13.99, 14.86, 17.06, 22.57] 

matnet_gap_final = [82.92, 181.68, None, None] # OOM for size 500 and 1000
glop_gap_final = [17.91, 27.96, 38.96, 47.77] 
avg_gap_gnn = [13.09, 13.04, 15.65, None]
avg_gap_final = [4.75, 7.82, 9.38, None]

# Create figure and subplots with larger size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot execution time on left axis with TRIPLED line widths and larger markers
l1 = ax1.plot(atsp_sizes, hetero_time_initial, marker='x', markersize=20,
             markeredgewidth=3, markerfacecolor='none',
             linestyle=(0, (5, 5)), linewidth=6,
             color='#1f77b4', label='Het-GAT-Concat + Greedy')

l2 = ax1.plot(atsp_sizes, hetero_time_final, marker='s', markersize=18,
             markeredgewidth=3, markerfacecolor='none',
             linestyle=(0, (5, 2, 1, 2)), linewidth=6,
             color='#d62728', label='Het-GAT-Concat + Edge Builder + 3-opt')

l3 = ax1.plot(atsp_sizes, matnet_time, marker='^', markersize=19,
             markeredgewidth=3, markerfacecolor='none',
             linestyle=(0, (5, 2, 1, 2, 1, 2)), linewidth=6,
             color='#ff7f0e', label='MatNet')

l4 = ax1.plot(atsp_sizes, glop_time, marker='o', markersize=18,
             markeredgewidth=3, markerfacecolor='none',
             linestyle=(0, (1, 2)), linewidth=6,
             color='#2ca02c', label='GLOP')

# Plot gap comparison on right axis with same enhanced styling
ax2.plot(atsp_sizes, avg_gap_gnn, marker='x', markersize=20,
        markeredgewidth=3, markerfacecolor='none',
        linestyle=(0, (5, 5)), linewidth=6,
        color='#1f77b4')

ax2.plot(atsp_sizes, avg_gap_final, marker='s', markersize=18,
        markeredgewidth=3, markerfacecolor='none',
        linestyle=(0, (5, 2, 1, 2)), linewidth=6,
        color='#d62728')

ax2.plot(atsp_sizes, matnet_gap_final, marker='^', markersize=19,
        markeredgewidth=3, markerfacecolor='none',
        linestyle=(0, (5, 2, 1, 2, 1, 2)), linewidth=6,
        color='#ff7f0e')

ax2.plot(atsp_sizes, glop_gap_final, marker='o', markersize=18,
        markeredgewidth=3, markerfacecolor='none',
        linestyle=(0, (1, 2)), linewidth=6,
        color='#2ca02c')

# Set y-axis to logarithmic scale for both subplots
ax1.set_yscale('log')
ax2.set_yscale('log')

# Enhanced formatting with larger fonts
for ax in (ax1, ax2):
    ax.set_xlabel('ATSP Size', fontsize=22)
    ax.set_xticks(atsp_sizes)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, linewidth=1.5, which='both')  # Add grid for both major and minor ticks
    
# Massive titles and labels
ax1.set_title('Execution Time Comparison (log scale)', fontsize=26, pad=25)
ax1.set_ylabel('Time (s)', fontsize=22)
ax1.set_ylim(1, 1000)  # Adjusted for log scale

ax2.set_title('Average Gap Comparison (log scale)', fontsize=26, pad=25)
ax2.set_ylabel('Avg Gap (%)', fontsize=22)
ax2.set_ylim(1, 1000)  # Adjusted for log scale

# Bold unified legend
handles = [l1[0], l2[0], l3[0], l4[0]]
ax2.legend(handles=handles, labels=[h.get_label() for h in handles],
          loc='upper right', fontsize=18, framealpha=0.9,
          handlelength=3, handletextpad=1)

plt.tight_layout(pad=5.0)  # More padding
plt.savefig('result_plot.pdf', bbox_inches='tight', dpi=300)
plt.close()