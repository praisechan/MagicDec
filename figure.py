import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('data.csv')

# 2. Fill down the prefill column
df['prefill'] = df['prefill'].ffill()
# df['prefill'] = df['prefill'].ffill()

# 3. Compute memory ratio
df['memory_ratio'] = df['budget(%)'] / 100.0

# 4. Pivot so each prefill size is its own line
pivot = df.pivot(index='memory_ratio',
                 columns='prefill',
                 values='acceptance rate')
# 4b. Reorder the columns to numeric order
order = ['8K','16K','32K','64K']
# order = ['','gov_report']
# order = ['llama3.1-8b','qwen2.5-7b','qwen2.5-14b']
pivot = pivot[order]

# 5. Plot
fig, ax = plt.subplots(figsize=(6,4))
for size in pivot.columns:
    ax.plot(pivot.index, pivot[size],
            marker='o',
            label=f'{size}')

ax.set_xlabel('Selection Ratio')
ax.set_ylabel('Acceptance Rate')
ax.set_title('Qwen2.5-14B - PG19')
ax.set_xticks(pivot.index)
ax.set_ylim(0.8, 1.0)
ax.grid(True)
ax.legend(title='Prefill')

plt.tight_layout()

# 6. Save to PNG
fig.savefig('qwen2.5-14b-pg19.png', dpi=300)

# (Optional) display on screen
# plt.show()