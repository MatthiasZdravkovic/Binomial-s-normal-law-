import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.patches import Ellipse

def generate_bivariate_normal(mean, cov, size):
    x, y = np.random.multivariate_normal(mean, cov, size).T
    return x, y

# Paramètres de la loi normale bidimensionnelle
µ_x = np.random.uniform(0, 10)
µ_y = np.random.uniform(0, 10)
mean = [µ_x, µ_y]  # Moyenne (mu_x, mu_y)
cov = [[3, 1.5],
       [1, 2  ]]  # Matrice de covariance

# Calculer les valeurs propres et les vecteurs propres de la matrice de covariance
valpropre, vectpropre = np.linalg.eigh(cov)

# Générer les données
size = 1000  # Taille de l'échantillon
x, y = generate_bivariate_normal(mean, cov, size)

# Dessiner les ellipses
probabilities = [0.3,0.5,0.9]
ellipses = []
for i, p in enumerate(probabilities):
    ellipse_radius = np.sqrt(2*valpropre*np.log(1/(1-p)))
    angle = np.degrees(np.arctan2(vectpropre[1, 0], vectpropre[0, 0]))
    ellipse = Ellipse(mean, width= 2*ellipse_radius[0], height= 2*ellipse_radius[1], angle=angle, edgecolor='red', facecolor='none')
    ellipses.append(ellipse)

# Créer le graph
fig, graph = plt.subplots()
graph.scatter(x, y, s=5)  # Points de dispersion
for ellipse in ellipses:
    graph.add_patch(ellipse)
graph.set_xlabel('X')
graph.set_ylabel('Y')
graph.set_title('Loi bidimensionnelle normale')
plt.grid(True)
plt.show()


# Partie estimateurs
def estimate(xx,yy,size2):
    µx_hat = np.mean(xx[:size2])
    µy_hat = np.mean(yy[:size2])
    cov_hat = np.cov(xx[:size2],yy[:size2])
    return (µx_hat,µy_hat,cov_hat)

fig, graphs = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
graphs = graphs.flatten()

size_ech=[1000,700,350,100]
colors = ['black', 'blue', 'green', 'yellow']
for j, graph1 in enumerate(graphs):
    µx_hat1,µy_hat1,cov_hat1 = estimate(x,y,size_ech[j])
    valpropre2, vectpropre2 = np.linalg.eigh(cov_hat1)
    ellipses2 = []
    for i, p in enumerate(probabilities):
        ellipse_radius2 = np.sqrt(2*valpropre*np.log(1/(1-p)))
        angle2 = np.degrees(np.arctan2(vectpropre2[1, 0], vectpropre2[0, 0]))
        ellipse2 = Ellipse(mean, width= 2*ellipse_radius2[0], height= 2*ellipse_radius2[1], angle=angle2, edgecolor=colors[j], facecolor='none')
        ellipses2.append(ellipse2)
    graph1.scatter(x, y, s=5)
    for ellipse1 in ellipses2:
        graph1.add_patch(ellipse1)
    for i, p in enumerate(probabilities):
        ellipse_radius = np.sqrt(2*valpropre*np.log(1/(1-p)))
        angle = np.degrees(np.arctan2(vectpropre[1, 0], vectpropre[0, 0]))
        ellipse = Ellipse(mean, width= 2*ellipse_radius[0], height= 2*ellipse_radius[1], angle=angle, edgecolor='red', facecolor='none')
        graph1.add_artist(ellipse)
        graph1.plot([], [], linestyle='solid', color='red')
    graph1.set_xlabel('X')
    graph1.set_ylabel('Y')
    graph1.set_title(f'Échantillon de taille {size_ech[j]}')
plt.show()
