import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Correcting and completing the input data for plotting.
data = {
    'Domain': ['Object Detection', 'Command Execution', 'Obstacle Navigation','Situation Awareness Success Rate'],
    'Human Surrogate(GPT-4-Turbo)':[40,40,0,40],
    'Human Surrogate(GPT-4o)':[60,60,20,40],
    'LA-RCS(GPT-4-Turbo)': [80, 80, 20, 60],
    'LA-RCS(GPT-4o)': [100, 100, 60, 100],
    'GPT-4-Turbo Avg Steps':  [6.8, 5.2, 9.4, 3.2],
    'GPT-4o Avg Steps': [5.2, 8.4, 8.0, 6.0]
}

# Creating a DataFrame with this data for easier plotting
df_revised = pd.DataFrame(data)
index = np.arange(len(df_revised['Domain']))

# Define pastel color palette
pastel_colors = [
    mcolors.CSS4_COLORS['lightblue'],
    mcolors.CSS4_COLORS['lightcoral'],
    mcolors.CSS4_COLORS['lightgreen'],
    mcolors.CSS4_COLORS['lightpink']
]

# Define bar width and positions to avoid overlap
bar_width = 0.15

# Success Rate Chart with pastel colors
plt.figure(figsize=(12, 6), dpi=500)

plt.bar(index , df_revised['LA-RCS(GPT-4-Turbo)'], width=bar_width, color=pastel_colors[0], label='LA-RCS(GPT-4-Turbo)')
plt.bar(index- bar_width, df_revised['LA-RCS(GPT-4o)'], width=bar_width, color=pastel_colors[1], label='LA-RCS(GPT-4o)')
plt.bar(index + 2 * bar_width, df_revised['Human Surrogate(GPT-4-Turbo)'], width=bar_width, color=pastel_colors[2], label='Human Surrogate(GPT-4-Turbo)')
plt.bar(index + 1 * bar_width, df_revised['Human Surrogate(GPT-4o)'], width=bar_width, color=pastel_colors[3], label='Human Surrogate(GPT-4o)')

plt.xticks(index + bar_width / 2, df_revised['Domain'])
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Domain')
plt.legend()

# Save before showing the plot
plt.savefig('result_success_pastel.png')
#plt.show()
