import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# --- Initialisierung ---
QUEUE_SIZE = 10
queue = deque(maxlen=QUEUE_SIZE)

# Listen zum Speichern der Ergebnisse für den Plot
iterations = []
signal_values = []
median_values = []
mean_values = []
trimmed_mean_values = []

iteration_counter = 0

# --- Simulationsphasen ---

# Phase 1: Stabiles niedriges Signal
print("Phase 1: Stabiles niedriges Signal wird aufgebaut...")
for i in range(30):
    iteration_counter += 1
    # Erzeuge einen Wert mit leichtem Rauschen um 10
    noise = random.uniform(-0.5, 0.5)
    current_signal = 10 + noise
    queue.append(current_signal)
    
    # Speichere die Werte für den Plot
    iterations.append(iteration_counter)
    signal_values.append(current_signal)
    median_values.append(np.median(queue))
    mean_values.append(np.mean(queue))
    # Getrimmter Mittelwert: Entferne die 20% größten und kleinsten Werte
    # Bei 10 Werten: Entferne den kleinsten und den größten
    trimmed_mean_values.append(np.mean(sorted(queue)[1:-1]))


# Phase 2: Ein starker Ausreißer
print("Phase 2: Ein starker Ausreißer wird eingefügt...")
iteration_counter += 1
outlier = 100
queue.append(outlier)

iterations.append(iteration_counter)
signal_values.append(outlier)
median_values.append(np.median(queue))
mean_values.append(np.mean(queue))
trimmed_mean_values.append(np.mean(sorted(queue)[1:-1]))


# Phase 3: Stabiles niedriges Signal, während Ausreißer durch die Queue wandert
print("Phase 3: Stabiles Signal, während Ausreißer die Queue verlässt...")
# 10 Schritte, damit der Ausreißer rausfällt + 20 weitere Schritte
for i in range(30):
    iteration_counter += 1
    noise = random.uniform(-0.5, 0.5)
    current_signal = 10 + noise
    queue.append(current_signal)
    
    iterations.append(iteration_counter)
    signal_values.append(current_signal)
    median_values.append(np.median(queue))
    mean_values.append(np.mean(queue))
    trimmed_mean_values.append(np.mean(sorted(queue)[1:-1]))


# Phase 4: Steigende Flanke und stabiles hohes Signal
print("Phase 4: Steigende Flanke und stabiles hohes Signal...")
for i in range(30):
    iteration_counter += 1
    noise = random.uniform(-0.5, 0.5)
    # Erzeuge einen Wert mit leichtem Rauschen um 25
    current_signal = 25 + noise
    queue.append(current_signal)
    
    iterations.append(iteration_counter)
    signal_values.append(current_signal)
    median_values.append(np.median(queue))
    mean_values.append(np.mean(queue))
    trimmed_mean_values.append(np.mean(sorted(queue)[1:-1]))

print("Simulation abgeschlossen. Plot wird erstellt.")

# --- Plotten der Ergebnisse ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(15, 8))

# Plotten der verschiedenen Werte
plt.plot(iterations, signal_values, 'o', color='grey', alpha=0.5, label='Originalsignal (Input)')
plt.plot(iterations, mean_values, 'r-', linewidth=2.5, label='Mittelwert (Mean)')
plt.plot(iterations, median_values, 'b-', linewidth=2.5, label='Median')
plt.plot(iterations, trimmed_mean_values, 'g--', linewidth=2.5, label='Getrimmter Mittelwert (20%)')

# Hinzufügen von Markierungen zur Verdeutlichung
plt.axvline(x=31, color='k', linestyle=':', label='Ausreißer-Event')
plt.axvline(x=61, color='purple', linestyle=':', label='Signal-Flanke Event')


# Titel und Beschriftungen
plt.title('Vergleich von Mittelwert, Median und getrimmtem Mittelwert', fontsize=16)
plt.xlabel('Iterationsschritte', fontsize=12)
plt.ylabel('Wert', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()