import carte_profondeur
import position_bac
import hauteurs_plantes
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import math
import numpy as np
import cv2 as cv
import csv
from itertools import zip_longest
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

file="/home/ecarnot/Téléchargements/Session 2023-03-16 09-02-40/uplot_103_1/uplot_103_camera_1_2_RGB.jpg"
left_path = file
id_image = left_path.split('camera_1')
right_path = 'camera_2'.join(id_image)
image_left = cv.imread(left_path)
image_right = cv.imread(right_path)

# Carte de profondeur
depth_image = carte_profondeur.workflow_carte_profondeur(image_left, image_right)
# Extraire la region du bac
haut, bas, gauche, droite, image_left_bac, image_right_bac = position_bac.contour_bac(image_left, image_right, seuil_small_obj)
image_cut = depth_image[haut:bas, gauche:droite]  # Carte de profondeur au dimensions du bac

mat_filtree = hauteurs_plantes.filtre_points_aberrants(depth_image)
matrice_filtree = depth_image.copy()
matrice_filtree[np.isinf(matrice_filtree)] = np.nan

height_map_clean = np.where(np.isfinite(matrice_filtree), depth_image, 5000)
height_map_clipped = np.clip(height_map_clean, a_min=None, a_max=6000)
plt.figure()
plt.hist(height_map_clipped.flatten(), bins=50, color='steelblue', edgecolor='black')
plt.title("Histogramme depth map; 5000=z_pts Inf; 6000=z_pts>6000")
plt.xlabel("Valeur")
plt.ylabel("Fréquence")


moyenne_actuelle = np.nanmean(mat_filtree)

        # Définir des bornes pour filtrer les points aberrants
        ecart_type = np.nanstd(mat_filtree)
        limite_inf = moyenne_actuelle - 5 * ecart_type
        limite_sup = moyenne_actuelle + 4 * ecart_type