# Revised costs from the LaTeX table (from the "Revised Costs(M)" column)
revised_costs = [
    1.7575,  # ATSP 100, Revision 3
    1.9619,  # ATSP 150, Revision 3
    2.0604,  # ATSP 250, Revision 3
    2.2355,  # ATSP 500, Revision 3
    2.3469,  # ATSP 1000, Revision 3
    1.7289,  # ATSP 100, Revision 6
    1.9134,  # ATSP 150, Revision 6
    2.0134,  # ATSP 250, Revision 6
    2.2111,  # ATSP 500, Revision 6
    2.3369,  # ATSP 1000, Revision 6
    1.6995,  # ATSP 100, Revision 9
    1.8906,  # ATSP 150, Revision 9
    1.9988,  # ATSP 250, Revision 9
    2.2037,  # ATSP 500, Revision 9
    2.3301   # ATSP 1000, Revision 9
]

# Average optimal costs provided
optimal_costs = {
    100: 1.5613,
    150: 1.6034,
    250: 1.5621,
    500: 1.5858,
    1000: 1.5768
}

# Function to calculate the percentage gap
def calculate_gap_percentage(optimal_cost, revised_cost):
    return ((revised_cost - optimal_cost) / optimal_cost) * 100

# List to store gaps for each ATSP size
gaps = []

# Iterate over ATSP sizes and calculate gaps
atsp_sizes = [100, 150, 250, 500, 1000]
revision_id = [3, 6, 9]
# Gather gaps for each revision level

for idx, size in enumerate(atsp_sizes):
    idxx = 0
    for revision in range(idx, 15, 5):  # Revising 3, 6, 9
        gap = calculate_gap_percentage(optimal_costs[size], revised_costs[revision])
        gaps.append((size, revision_id[idxx], gap))
        idxx += 1

# Print gaps
for size, revision, gap in gaps:
    print(f"ATSP size {size}, Revision {revision}: Gap = {gap:.2f}%")
