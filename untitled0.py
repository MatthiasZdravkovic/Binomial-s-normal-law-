# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:14:12 2023

@author: Matthias
"""

import numpy as np
import matplotlib.pyplot as plt

def density_Z(x, y, h, k, a, b, theta, K):
    term1 = ((x - h) * np.cos(theta) + (y - k) * np.sin(theta))**2 / (2*a*np.log(1/(2*np.pi*K*np.sqrt(a*b))))
    term2 = (-(x - h) * np.sin(theta) + (y - k) * np.cos(theta))**2 / (2*a*np.log(1/(2*np.pi*K*np.sqrt(a*b))))
    return term1 + term2

# Paramètres de l'ellipse
h = 0.5  # Centre x
k = 0.5  # Centre y
a = 2.0  # Demi-longueur de l'axe principal
b = 0.5  # Demi-longueur de l'axe secondaire
theta = np.pi/4  # Angle de rotation
K = 0.5  # Facteur de normalisation

# Génération des points
num_points = 1000
x = np.random.normal(0, 1, num_points)
y = np.random.normal(0, 1, num_points)
density = density_Z(x, y, h, k, a, b, theta, K)

# Probabilités des ellipses d'isodensité
p_values = [0.1, 0.3, 0.5, 0.7, 0.9]

# Tracé des ellipses d'isodensité
fig, ax = plt.subplots()
ax.scatter(x, y, c=density, cmap='viridis')
for p in p_values:
    threshold = np.percentile(density, p*100)
    ellipse = plt.Ellipse((h, k), a*2*threshold, b*2*threshold, theta*180/np.pi, edgecolor='red', facecolor='None')
    ax.add_patch(ellipse)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Échantillon de points selon la loi Z et ellipses d\'isodensité')
plt.colorbar(label='Densité')
plt.axis('equal')
plt.show()
