import matplotlib.pyplot as plt
import numpy as np

models = ["Gemma2", "TinyLlama", "Mixtral 8x22", "Mistral Latest", "OpenChat", "Llama 3.3"]
time_execution = [68.0, 63.9, 95.2, 66.0, 76.9, 99.0]
memory_usage = [8714, 1622, 41398, 5772, 5768, 44014]

bar_width = 0.4
x = np.arange(len(models))

fig, ax1 = plt.subplots(figsize=(10, 6))

bars1 = ax1.bar(x - bar_width/2, time_execution, bar_width, label='Temps d\'exécution (s)', color='blue')
ax1.set_ylabel("Temps d'exécution (secondes)", color='blue')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha="right")
ax1.set_xlabel("Modèles")
ax1.set_title("Benchmark des modèles IA")
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + bar_width/2, memory_usage, bar_width, label='Mémoire utilisée (MiB)', color='red', alpha=0.6)
ax2.set_ylabel("Mémoire utilisée (MiB)", color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.tight_layout()
plt.show()