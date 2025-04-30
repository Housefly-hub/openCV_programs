import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from numpy.linalg import norm
from math import degrees, acos

# Full vectors
A = np.array([1,2,3,4,5,6,7,8,9,10])
B = np.array([1,3,5,7,9,7,5,3,1,0])

# Cosine Similarity
cos_sim = cosine_similarity([A], [B])[0][0]
angle_deg = degrees(acos(cos_sim))

# Euclidean Distance
euclid_dist = euclidean(A, B)

# Output results
print(f"Cosine Similarity (A, B): {cos_sim:.4f}")
print(f"Angle between A and B: {angle_deg:.2f}°")
print(f"Euclidean Distance (A, B): {euclid_dist:.4f}")

# Project to 2D for visualization
A_2D = A[:2]
B_2D = B[:2]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot 1: Cosine Similarity (Angle)
axs[0].quiver(0, 0, A_2D[0], A_2D[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector A')
axs[0].quiver(0, 0, B_2D[0], B_2D[1], angles='xy', scale_units='xy', scale=1, color='green', label='Vector B')
axs[0].set_xlim(0, max(A_2D[0], B_2D[0]) + 2)
axs[0].set_ylim(0, max(A_2D[1], B_2D[1]) + 2)
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')
axs[0].set_title(f'Cosine Similarity\nAngle: {angle_deg:.2f}°')
axs[0].legend()
axs[0].grid()
axs[0].set_aspect('equal', adjustable='box')

# Plot 2: Euclidean Distance
axs[1].quiver(0, 0, A_2D[0], A_2D[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector A')
axs[1].quiver(0, 0, B_2D[0], B_2D[1], angles='xy', scale_units='xy', scale=1, color='green', label='Vector B')
axs[1].plot([A_2D[0], B_2D[0]], [A_2D[1], B_2D[1]], 'r--', label='Euclidean Distance')
axs[1].set_xlim(0, max(A_2D[0], B_2D[0]) + 2)
axs[1].set_ylim(0, max(A_2D[1], B_2D[1]) + 2)
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Y-axis')
axs[1].set_title(f'Euclidean Distance\nDistance: {euclid_dist:.2f}')
axs[1].legend()
axs[1].grid()
axs[1].set_aspect('equal', adjustable='box')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("images/similarity_dissimilarity_plot.png", dpi=300)
plt.show()
