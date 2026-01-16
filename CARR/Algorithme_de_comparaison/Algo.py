import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 14))

# Algorithm data
algorithms = ['RL\n(DQN)', 'NN\n(LSTM)', 'k-NN', 'SVM', 'DT/RF', 'K-Means\nGMM', 'HMM', 'Semi-sup', 'NB/Rules']

# Normalized metrics (scores out of 10)
performance_scores = [9, 8, 5, 7, 8, 6, 7, 7, 5]
interpretability_scores = [2, 3, 8, 5, 9, 6, 7, 6, 10]
scalability_scores = [4, 5, 3, 3, 7, 8, 5, 6, 10]
training_time_scores = [2, 4, 10, 5, 7, 8, 6, 6, 9]
inference_time_scores = [7, 6, 3, 6, 8, 9, 6, 7, 10]
robustness_scores = [8, 6, 4, 7, 8, 5, 6, 6, 6]

# GRAPH 1: Radar Chart
ax1 = plt.subplot(2, 3, 1, projection='polar')
categories = ['Performance', 'Interpretability', 'Scalability', 'Training Time', 'Inference Time', 'Robustness']
N = len(categories)
algo_indices = [0, 1, 4]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
algo_names_radar = ['RL (DQN)', 'NN (LSTM)', 'DT/RF']
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for idx, algo_idx in enumerate(algo_indices):
    values = [performance_scores[algo_idx], interpretability_scores[algo_idx], scalability_scores[algo_idx],
              training_time_scores[algo_idx], inference_time_scores[algo_idx], robustness_scores[algo_idx]]
    values += values[:1]
    ax1.plot(angles, values, 'o-', linewidth=2, label=algo_names_radar[idx], color=colors[idx])
    ax1.fill(angles, values, alpha=0.15, color=colors[idx])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, size=9)
ax1.set_ylim(0, 10)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax1.set_title('Multi-criteria Comparison', fontsize=11, fontweight='bold', pad=20)
ax1.grid(True, linestyle='--', alpha=0.6)

# GRAPH 2: Bar Chart
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(algorithms))
width = 0.35
bars1 = ax2.bar(x - width/2, performance_scores, width, label='Performance', color='#FF6B6B', alpha=0.8)
bars2 = ax2.bar(x + width/2, interpretability_scores, width, label='Interpretability', color='#4ECDC4', alpha=0.8)
ax2.set_xlabel('Algorithms', fontweight='bold')
ax2.set_ylabel('Score (0-10)', fontweight='bold')
ax2.set_title('Performance vs Interpretability', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=8)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# GRAPH 3: Scatter Plot
ax3 = plt.subplot(2, 3, 3)
scatter = ax3.scatter(scalability_scores, inference_time_scores, s=[p*50 for p in performance_scores],
                      c=performance_scores, cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1.5)
for i, txt in enumerate(algorithms):
    ax3.annotate(txt.replace('\n', ' '), (scalability_scores[i], inference_time_scores[i]),
                fontsize=7, ha='center')
ax3.set_xlabel('Scalability', fontweight='bold')
ax3.set_ylabel('Inference Time', fontweight='bold')
ax3.set_title('Scalability vs Inference Time', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Performance')

# GRAPH 4: Heatmap
ax4 = plt.subplot(2, 3, 4)
metrics_data = np.array([performance_scores, interpretability_scores, scalability_scores,
                        training_time_scores, inference_time_scores, robustness_scores])
im = ax4.imshow(metrics_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)
ax4.set_xticks(np.arange(len(algorithms)))
ax4.set_yticks(np.arange(len(categories)))
ax4.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=8)
ax4.set_yticklabels(categories, fontsize=9)
for i in range(len(categories)):
    for j in range(len(algorithms)):
        ax4.text(j, i, int(metrics_data[i, j]), ha="center", va="center", fontsize=8, fontweight='bold')
ax4.set_title('Score Matrix', fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax4)

# GRAPH 5: Grouped Bar
ax5 = plt.subplot(2, 3, 5)
bars3 = ax5.bar(x - width/2, training_time_scores, width, label='Training Time', color='#FF8C42', alpha=0.8)
bars4 = ax5.bar(x + width/2, inference_time_scores, width, label='Inference Time', color='#9B59B6', alpha=0.8)
ax5.set_xlabel('Algorithms', fontweight='bold')
ax5.set_ylabel('Time Score', fontweight='bold')
ax5.set_title('Time Comparison', fontsize=11, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=8)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# GRAPH 6: Global Score
ax6 = plt.subplot(2, 3, 6)
weights = {'performance': 0.30, 'interpretability': 0.15, 'scalability': 0.20,
         'training_time': 0.10, 'inference_time': 0.15, 'robustness': 0.10}
global_score = []
for i in range(len(algorithms)):
    score = (performance_scores[i] * weights['performance'] + interpretability_scores[i] * weights['interpretability'] +
             scalability_scores[i] * weights['scalability'] + training_time_scores[i] * weights['training_time'] +
             inference_time_scores[i] * weights['inference_time'] + robustness_scores[i] * weights['robustness'])
    global_score.append(score)

sorted_indices = np.argsort(global_score)[::-1]
sorted_algorithms = [algorithms[i] for i in sorted_indices]
sorted_global_scores = [global_score[i] for i in sorted_indices]
bar_colors = ['#2ECC71' if s >= 7 else '#F39C12' if s >= 5.5 else '#E74C3C' for s in sorted_global_scores]
bars5 = ax6.barh(sorted_algorithms, sorted_global_scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
ax6.set_xlabel('Weighted Global Score', fontweight='bold')
ax6.set_title('Global Ranking', fontsize=11, fontweight='bold')
ax6.set_xlim(0, 10)
ax6.grid(axis='x', alpha=0.3)
for i, score in enumerate(sorted_global_scores):
    ax6.text(score + 0.1, i, f'{score:.2f}', va='center', fontweight='bold')

fig.suptitle('Comparative Analysis of ML Algorithms for SDN', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig('sdn_algorithms_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Graphs saved: sdn_algorithms_analysis.png")
plt.show()

# Summary table
df = pd.DataFrame({
    'Algorithm': algorithms,
    'Performance': performance_scores,
    'Interpretability': interpretability_scores,
    'Scalability': scalability_scores,
    'Global Score': [f"{s:.2f}" for s in global_score]
})
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(df.to_string(index=False))
print("="*80)