# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. Load the data
# df = pd.read_csv('data.csv')

# # 2. Fill down the prefill column
# df['prefill'] = df['prefill'].ffill()

# # 3. Compute memory ratio
# df['memory_ratio'] = df['budget(%)'] / 100.0

# # 4. Pivot so each prefill size is its own line
# pivot = df.pivot(index='memory_ratio',
#                  columns='prefill',
#                  values='acceptance rate')
# # 4b. Reorder the columns to numeric order
# order = ['8K','16K','32K','64K']
# pivot = pivot[order]

# # 5. Plot
# fig, ax = plt.subplots(figsize=(6,4))
# for size in pivot.columns:
#     ax.plot(pivot.index, pivot[size],
#             marker='o',
#             label=f'{size}')

# ax.set_xlabel('Selection Ratio')
# ax.set_ylabel('Acceptance Rate')
# ax.set_title('Acceptance Rate vs. Selection Ratio')
# ax.set_xticks(pivot.index)
# ax.set_ylim(0.8, 1.0)
# ax.grid(True)
# ax.legend(title='Prefill')

# plt.tight_layout()

# # 6. Save to PNG
# fig.savefig('acceptance_vs_memory.png', dpi=300)

# # (Optional) display on screen
# # plt.show()
import os, csv
CSV_PATH = "/home/juchanlee/MagicDec/output/llama-3.1-8B_PG19_acceptance_rates.csv"
# if the file doesn't yet exist, write the header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix_len", "draft_budget", "gamma", "accept_rate"])
        
# append to CSV
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        1,2,3,4
    ])
