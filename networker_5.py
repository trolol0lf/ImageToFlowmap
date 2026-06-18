 #-uses:
#-clustering for meanshift
#-per-cluster-skeleton
# ->contours
# ->direction of contours should be similar
# ->contour to spline
# ->spline searches border
# ->fill from last borderpoints to next
from __future__ import annotations 

logger = None # will be set from parent script -> ImageToFlowmapManager

from typing import Literal, Tuple, Optional, List, get_args, TYPE_CHECKING
import cv2, os, pickle

import numpy as np

from skimage import morphology, segmentation, color
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

if TYPE_CHECKING:
    from ImageToFlowmap.ImageToFlowmapManager import FlowMapGenFiles
from settings import *
import flowmap_vektor_length_unionizer as fvlu
from scipy.ndimage import label, center_of_mass, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
#from scikit-image import restoration import denoise_tv_chambolle
from collections import Counter
import matplotlib.pyplot as plt

from random import randint
from multiprocessing import Queue

from types import MethodType

Init_TimeGroup("processfiles_file")

neighmodes = {
    "Neigh8" :      [(-1,-1), (-1,0), (-1,1),
                     (0,-1),          (0,1),
                     (1,-1),  (1,0),  (1,1)],
    "Neigh4-90" :   [(-1,0),(0,-1),(0,1),(1,0)]
}

currfile = None
currh = 0
currw = 0

# -----------------------------
# Konfig – knappe Defaults
# -----------------------------
SMOOTHING_S = 2.0          # Spline-Glättung (0 = Interpolation)
SAMPLES_PER_PATH = 400     # Spline-Samples pro Pfad (Arclength-uniform, s.u.)
GRID_RES = 2.0             # Rasterabstand fürs Vektorfeld (Pixel/Map-Einheiten)
KERNEL_SIGMA_FACT = 0.35   # σ ~ KERNEL_SIGMA_FACT * lokale Breite
NEIGHBOR_RADIUS_FACT = 2.0 # Nachbarschaftsradius ~ NEIGHBOR_RADIUS_FACT * σ
BASE_WIDTH = 8.0           # Basisbreite, falls keine eigene Breite pro Pfad
BASE_DEPTH = 1.0           # Basistiefe, falls keine eigene Tiefe pro Pfad

# -----------------------------------
# Hilfsfunktionen: Geometrie & Spline
# -----------------------------------

# Save helper für Zwischenschritte
def get_add_grad(field, fig, ax, targetgradientcount = 1, currgradientcount = 0):
    img = field
    if targetgradientcount != currgradientcount:
        img = np.gradient(field,axis=0)
        ax[currgradientcount].imshow(img, cmap='magma')
        #ax[2].plot(ridge_response[:, 1], ridge_response[:, 0], 'ro', markersize=3)
        ax[currgradientcount].set_title(f'Gradient #{currgradientcount}')
        if targetgradientcount != currgradientcount+1:
            img = get_add_grad(field=img, fig=fig, ax=ax, targetgradientcount=targetgradientcount, currgradientcount=currgradientcount+1)
    return img

def smooth_mask_by_mode(mask, mode = Literal["Morph-Opening", "Gaussian"], MorphDiskIndex = 1, GaussSigma = 1.2):
    match mode:
        case "Morph-Opening":
            return morphology.opening(mask, morphology.disk(MorphDiskIndex))
        case "Gaussian":
            return gaussian_filter(mask.astype(float), sigma=GaussSigma / RefImagePropertySizeFactor) > 0.0

def weird_splitted_skeleton(mask, ShowSteps = False, masksmoothmode : List[Literal["Morph-Opening", "Gaussian"]] = []):
        # Schritt 1: Distance Transform berechnen
    # ---------------------------------------
    # Visualisierung der Substeps
    
    #
    blur_mask = mask.copy()
    for modes in masksmoothmode:
        blur_mask = smooth_mask_by_mode(blur_mask, modes, MorphDiskIndex = 1, GaussSigma = 1.2)

    blur_mask[~mask] = False
    
    skeleton, dist  = morphology.medial_axis(blur_mask, return_distance=True)
    #skeleton = morphology.thin(skeleton)
    
    if ShowSteps:
        # Original Maske
        fig, ax = plt.subplots(1, 3, figsize=(18, 8))
        ax[0].imshow(mask, cmap='gray')
        ax[0].contour(skeleton, colors='cyan')
        ax[0].set_title('Cluster Maske + Skeleton')


        # Blurred Mask 
        ax[1].imshow(blur_mask, cmap='magma')
        ax[1].set_title('Blurred Mask')

        # Distance Transform
        ax[2].imshow(dist, cmap='magma')
        ax[2].set_title('Distance Map')

        plt.tight_layout()
        plt.show(block=True)
        #print(f"FAFAFF")
    return skeleton

def plot_paths_with_arrows(pointlistlist, bgimage, PointsAreYX = True, ShowPaths = False):
    """
    Visualisiert eine Punktliste mit Pfeilen.
    
    Args:
        points (list of (int,int)): Liste von (x,y)-Koordinaten.
        step (int): Alle 'step' Punkte wird ein Pfeil gezeichnet.
        out_file (str): Dateiname für die Ausgabe (PNG mit Transparenz).
        img_size (tuple): Größe des Ausgabebildes in Pixeln.
    """

    rbgimage = bgimage.copy()
    dt8, bs8 = get_datatype(8)
    rgbimage = ensure_save_flowmap_format(rbgimage, True, True, 8, False)
    img_size = rgbimage.shape
    h,w = img_size[0], img_size[1]
    colors = np.random.rand(len(pointlistlist), 4)
    colors = colors * 0.5 + 0.5
    colors[:,3] = 1.0

    # gewünschte Maximalgröße in Pixel (z. B. FullHD Monitor)
    max_w, max_h = 1920, 1080  
    # aktuelle DPI von matplotlib
    dpi = plt.rcParams['figure.dpi']

    # Figure size in Inches limitieren
    fig_w = min(w, max_w) / dpi
    fig_h = min(h, max_h) / dpi

    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

    # Hintergrund transparent
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0,0,0,0))
    ax.imshow(rgbimage, extent=[0, w, 0, h], origin='lower')

    for j, points in enumerate(pointlistlist):
        if len(points) == 0: continue
        if PointsAreYX: points = [p[::-1] for p in points]
        step = int((len(points) / (len(points)**0.5)) + 1)
        points = np.array(points)
        

        # Pfad zeichnen
        ax.plot(points[:,0], points[:,1], color=colors[j])

        # Pfeile einfügen
        for i in range(0, len(points)-1, step):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=colors[j], lw=2)
            )

    # Achsen ausblenden
    ax.axis("off")
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    #fig.show()
    if 1 == 0:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=dt8)
        buf = buf.reshape((h, w, 4))
        # ARGB -> RGBA
        dt16, bs16 = get_datatype(16)
        img = buf[:, :, [1,2,3,0]]
        img = (img / bs8).astype(dt16) * bs16
    else:
        img = np.array(fig.canvas.renderer.buffer_rgba())
        dt16, bs16 = get_datatype(16)
        img = (img / bs8 * bs16).astype(dt16)

    if ShowPaths:
        fig.set_size_inches(fig_w,fig_h)
        plt.show(block=True)
    return img

def rdp(points, epsilon):
    """
    Ramer–Douglas–Peucker Vereinfachung für (y,x)-Punkte.
    points: np.ndarray [N,2]
    epsilon: float (Pixel)
    """
    if len(points) < 3:
        return points

    # gerichteter Abstand Punkt -> Linie (start,end)
    def perp_dist(p, a, b):
        if np.allclose(a, b):
            return np.linalg.norm(p - a)
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)

    start, end = points[0], points[-1]
    dmax, idx = 0.0, 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], start, end)
        if d > dmax:
            idx, dmax = i, d

    if dmax > epsilon:
        left = rdp(points[: idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([start, end])

def catmull_rom_to_cubic_bezier(P0, P1, P2, P3, alpha=0.5):
    """
    Konvertiert ein Catmull-Rom Segment (P1->P2) in Bézier-Kontrollpunkte (B0..B3).
    'alpha' steuert die Parametrisierung (0.5 = centripetal i.d.R. stabiler als uniform).
    Formel: wir nutzen die (gängige) Approximation mit 1/6-Faktoren.
    """
    # Für robuste Tension nehmen wir die "centripetal" Approx:
    # B0 = P1
    # B3 = P2
    # B1 = P1 + (P2 - P0)/6
    # B2 = P2 - (P3 - P1)/6
    B0 = P1
    B3 = P2
    B1 = P1 + (P2 - P0) / 6.0
    B2 = P2 - (P3 - P1) / 6.0
    return np.stack([B0, B1, B2, B3], axis=0)

def sample_cubic_bezier(B, n_samples):
    """
    B: (4,2) Kontrollpunkte (y,x)
    Gibt:
      pts: (n,2) Punkte
      d1 : (n,2) 1. Ableitung (Tangente, unnormiert)
    """
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    # Bernstein-Polynome
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t ** 2
    b3 = t ** 3

    pts = (B[0] * b0[:, None] +
           B[1] * b1[:, None] +
           B[2] * b2[:, None] +
           B[3] * b3[:, None])

    # Ableitung: 3 * [ (B1 - B0)*(1-t)^2 + 2*(B2 - B1)*(1-t)*t + (B3 - B2)*t^2 ]
    db0 = -3 * (1 - t) ** 2
    db1 = 3 * (1 - t) ** 2 - 6 * (1 - t) * t
    db2 = 6 * (1 - t) * t - 3 * t ** 2
    db3 = 3 * t ** 2

    # Alternative kompakte Form:
    d1 = (B[0] * db0[:, None] +
          B[1] * db1[:, None] +
          B[2] * db2[:, None] +
          B[3] * db3[:, None])
    return pts, d1

def build_bezier_samples_from_polyline(poly, rdp_eps=1.5, samples_per_seg=32, max_ctrl_pts=20):
    """
    poly: np.ndarray [N,2] (y,x)
    - vereinfacht per RDP
    - baut Catmull-Rom Spline über vereinfachte Punkte
    - konvertiert jedes Segment in kubische Bézier & sampelt Punkte + Tangenten
    """
    if len(poly) < 2:
        return np.empty((0,2), dtype=np.float32), np.empty((0,2), dtype=np.float32)

    simp = rdp(poly, rdp_eps)
    # Optional weitere Reduktion auf max_ctrl_pts
    if len(simp) > max_ctrl_pts and 1 == 0:
        idx = np.linspace(0, len(simp) - 1, max_ctrl_pts).astype(int)
        simp = simp[idx]

    if len(simp) < 2:
        return simp.astype(np.float32), np.gradient(simp.astype(np.float32), axis=0)

    # Für Catmull-Rom brauchen wir 4-Punkt-Fenster; Enden duplizieren
    P = np.vstack([simp[0], simp, simp[-1]])
    #P = np.vstack([simp[0:1], simp, simp[-1:]])
    # Noch ein Duplikat für rechtes Ende
    P = np.vstack([P, P[-1]])

    all_pts = []
    all_tan = []
    for i in range(1, len(P) - 2):
        P0, P1, P2, P3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        B = catmull_rom_to_cubic_bezier(P0, P1, P2, P3)
        pts, d1 = sample_cubic_bezier(B, samples_per_seg)
        all_pts.append(pts)
        all_tan.append(d1)

    pts = np.vstack(all_pts).astype(np.float32)
    tan = np.vstack(all_tan).astype(np.float32)

    pts[-samples_per_seg:] = simp[-1]
    tan[-samples_per_seg:] = 0
    return pts, tan

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)



# ============================================================
# Hauptpipeline
# ============================================================

def flowmap_from_skeletons(
    polylines,            # List[List[(y,x), ...]]
    clustermask,          # np.ndarray (H,W) bool
    rdp_eps=1.5,
    samples_per_seg=32,
    max_ctrl_pts=20,
    skeleton_stamp_radius=1,  # Radius (Pixel) beim Rasterisieren des Skeletons
    skeleton_mask = None,
    DebugOutputImages = False,
    ShowPaths = False
):
    H, W = clustermask.shape
    # 1) Glätten/Vereinfachen + Bézier-Sampling sammeln
    all_pts = []
    all_tan = []
    PrintArrowMap = AccessMngr.can_access(["processfiles", "flowmapgen", "arrowmap"])
    if ShowPaths or PrintArrowMap:
        img = plot_paths_with_arrows(pointlistlist=polylines, bgimage=clustermask, PointsAreYX=True, ShowPaths=ShowPaths)
        if PrintArrowMap: SaveFlowmap(outputfolder, file.name, file.format, img, "_arrowsb4_", "lol", True)
    polylinemaxlength = [len(poly) for poly in polylines]
    polylinemaxlength.sort()
    polylinemaxlength = polylinemaxlength[-1]
    for poly in polylines:
        p = np.asarray(poly, dtype=np.float32)
        pts, tan = build_bezier_samples_from_polyline(
            p, rdp_eps=rdp_eps, samples_per_seg=int(np.ceil(samples_per_seg*RefImagePropertySizeFactor*len(poly)/polylinemaxlength)), max_ctrl_pts=int(np.ceil(max_ctrl_pts*RefImagePropertySizeFactor*len(poly)/polylinemaxlength))
        )
        if len(pts):
            all_pts.append(pts)
            all_tan.append(tan)

    if not all_pts:
        # Fallback: leere Flowmap
        flow = np.zeros((H, W, 4), dtype=np.uint8)
        #flow[..., 3] = 255
        return flow

    if ShowPaths or PrintArrowMap:  
        img = plot_paths_with_arrows(pointlistlist=all_pts, bgimage=clustermask, PointsAreYX=True, ShowPaths=ShowPaths)
        if PrintArrowMap: SaveFlowmap(outputfolder, file.name, file.format, img, "_arrowsAF_", "lol", True)

    skel_xy = np.vstack(all_pts)          # (N,2) (y,x)
    tan_xy  = np.vstack(all_tan)          # (N,2) (dy,dx)
    tan_xy  = normalize(tan_xy)

    # 2) KDTree für nächstgelegenen Skelettpunkt (per Pixel)
    tree = cKDTree(skel_xy[:, ::-1])  # KDTree auf (x,y) anlegen

    # 3) Skeleton-Maske rastern (für Distanz zu Skeleton)
    if type(skeleton_mask) == type(None):   ######TODO: Finding out why the hell we should be working with a skeleton mask!?
        skeleton_mask = np.zeros((H, W), dtype=bool)
        yy = np.clip(np.round(skel_xy[:, 0]).astype(int), 0, H - 1)
        xx = np.clip(np.round(skel_xy[:, 1]).astype(int), 0, W - 1)
        skeleton_mask[yy, xx] = True
        # Optional: "Stamp" für Lückenrobustheit
        if skeleton_stamp_radius > 0:
            r = skeleton_stamp_radius
            y_idx, x_idx = np.where(skeleton_mask)
            for y, x in zip(y_idx, x_idx):
                y0, y1 = max(0, y - r), min(H, y + r + 1)
                x0, x1 = max(0, x - r), min(W, x + r + 1)
                skeleton_mask[y0:y1, x0:x1] = True
    
    cy, cx = np.where(clustermask)
    ncy, ncx = np.where(~clustermask)
    # 4) Distance Transforms
    #   - Abstand zur Clusterrand (innen): Distanz im True-Gebiet zu False -> EDT(clustermask)
    dist_to_edge = distance_transform_edt(clustermask)#.astype(np.float64)
    #dist_to_edge = gaussian_filter(dist_to_edge, sigma=2.0)
    dist_to_edge[ncy, ncx] = 0    
    if DebugOutputImages: SaveFlowmap(outputfolder, file.name, file.format, dist_to_edge, "_disttoedge_", "lol", True, True)
    
    distancemode : Literal["dist_to_skel", "dist_to_edge"] = "dist_to_edge"
    
    if distancemode == "dist_to_skel": #since the skeleton is not perfect, we probably want to work with blurred dist to edge
        distancemap = distance_transform_edt(~skeleton_mask,)#.astype(np.float64)    
        distancemap[ncy, ncx] = 0  #remove cluster
        maxdisttoskel = distancemap[cy, cx].max()
        distancemap[cy, cx] = 1 - (distancemap[cy, cx] / maxdisttoskel)  #inversion -> inverted dist to skel = isotachen 
    elif distancemode == "dist_to_edge":
        distancemap = distance_transform_edt(clustermask)
        distancemap[ncy, ncx] = 0
        maxdisttoskel = distancemap[cy, cx].max()
        distancemap[cy, cx] = distancemap[cy, cx] / maxdisttoskel

    if DebugOutputImages: SaveFlowmap(outputfolder, file.name, file.format, distancemap, "_distancemap_", "lol", True, True)
    
    #Blur
    distancemap = gaussian_filter(distancemap, sigma=2.0)
    if DebugOutputImages: SaveFlowmap(outputfolder, file.name, file.format, distancemap, "_distancemap_blur_", "lol", True, True)
    
    #Mask
    distancemap[ncy, ncx] = 0
    if DebugOutputImages: SaveFlowmap(outputfolder, file.name, file.format, distancemap, "_distancemap_mask_", "lol", True, True)

    def mask_to_graph(mask):
        h, w = mask.shape
        idx_map = -np.ones_like(mask, dtype=int)
        idx_map[mask] = np.arange(mask.sum())

        rows, cols, data = [], [], []
        for y in range(h):
            for x in range(w):
                if not mask[y, x]:
                    continue
                i = idx_map[y, x]
                for dy, dx in [(1,0), (0,1), (-1,0), (0,-1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                        j = idx_map[ny, nx]
                        rows.append(i); cols.append(j); data.append(1.0)
        return csr_matrix((data, (rows, cols)), shape=(mask.sum(), mask.sum())), idx_map
    
    def get_dists_mask(mask, return_predecessors = True):
        graph, idx_map = mask_to_graph(mask)
        # 2. Geodätische Distanzen statt euklidischer
        start_idx = idx_map[y0, x0]
        dists, predecessors = dijkstra(csgraph=graph, directed=False, indices=start_idx, return_predecessors=return_predecessors)
        return dists, predecessors

    def get_tans_by_mask(mask, maxdist = 150):
        dists, idxs = get_dists_mask(mask, True)
        mask = dists < maxdist * RefImagePropertySizeFactor   # nur Nachbarn im Umkreis von k px
        weights = None
        weightmode = True
        nearest_tans = tan_xy[idxs]              # (M,3,2)
        if weightmode:
            weights = np.where(mask, 1.0/(dists+1e-6), 0.0)
            if len(weights.shape) > 1:
                row_sums = weights.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                weights /= row_sums
            else:
                row_sums = weights.sum()
                row_sums[row_sums == 0] = 1.0
                weights /= weights.sum()
            avg_tan = (nearest_tans * weights[...,None]).sum(axis=1)
        else:
            avg_tan = nearest_tans.sum(axis=1)
        #Test: normalize, cuz we always have flow strength 1 around here!
        avg_tan = avg_tan / np.clip(np.linalg.norm(avg_tan, axis=1, keepdims=True), 1e-8, None)
        return dists, idxs, nearest_tans, avg_tan
    
    def filter_tans_by_opposing_dir(nearest_tans, weights=None, angle_thresh=np.pi/3):
        """
        nearest_tans: (M, k, 2)
        weights: (M, k) oder None
        """
        # Normieren
        norms = np.linalg.norm(nearest_tans, axis=2, keepdims=True) + 1e-8
        unit_tans = nearest_tans / norms   # (M, k, 2)

        # Mehrheitsrichtung pro Punkt
        if weights is None:
            avg_dirs = unit_tans.mean(axis=1)  # (M, 2)
        else:
            avg_dirs = (unit_tans * weights[..., None]).sum(axis=1)  # (M, 2)
        avg_dirs /= np.linalg.norm(avg_dirs, axis=1, keepdims=True) + 1e-8

        # Winkel via Skalarprodukt
        cos_sim = np.einsum("mki,mi->mk", unit_tans, avg_dirs)  # (M, k)
        mask = cos_sim > np.cos(angle_thresh)                   # (M, k)

        # Gefilterte Tans + Weights (werden "genullt")
        filtered_tans = nearest_tans * mask[..., None]
        if weights is not None:
            filtered_weights = weights * mask
        else:
            filtered_weights = None

        return filtered_tans, filtered_weights
    
    def get_tans_by_tree(tree, coords, k, maxdist = 150.0):
        dists, idxs = tree.query(coords, k=k)   # statt k=1
        mask = dists < maxdist * RefImagePropertySizeFactor   # nur Nachbarn im Umkreis von k px
        weights = None
        weightmode = True
        nearest_tans = tan_xy[idxs]              # (M,3,2)
        weights = np.where(mask, 1.0/(dists+1e-6), 0.0)
        if 1 == 0:
            nearest_tans, weights = filter_tans_by_opposing_dir(nearest_tans=nearest_tans, weights=weights)
        if weightmode:
            if len(weights.shape) > 1:
                row_sums = weights.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                weights /= row_sums
            else:
                row_sums = weights.sum()
                row_sums[row_sums == 0] = 1.0
                weights /= weights.sum()
            avg_tan = (nearest_tans * weights[...,None]).sum(axis=1)
        else:
            avg_tan = nearest_tans.sum(axis=1)
        #Test: normalize, cuz we always have flow strength 1 around here!
        avg_tan = avg_tan / np.clip(np.linalg.norm(avg_tan, axis=1, keepdims=True), 1e-8, None)
        return dists, idxs, nearest_tans, avg_tan  

    # 5) Pixelkoordinaten für Cluster sammeln und nächste Tangente holen
    cy, cx = np.where(clustermask)
    coords = np.stack([cx, cy], axis=1)  # (M,2) -> (x,y)
    
    if 1 == 1:
        dists, idxs, nearest_tans, avg_tan = get_tans_by_tree(tree=tree, coords=coords, k=int(64*RefImagePropertySizeFactor), maxdist=int(125*RefImagePropertySizeFactor))
    else:
        dists, idxs, nearest_tans, avg_tan = get_tans_by_mask(mask=clustermask, maxdist=int(125*RefImagePropertySizeFactor))
    
    avgtanimg = np.zeros((H,W,2), dtype=np.float64)
    avgtanimg[cy,cx] = avg_tan
    for steps in range(4):  #
        avgtanimg = fvlu.masked_median_blur(avgtanimg, clustermask, ksize=6, ignore_color=(0,0)) # 4
    avg_tan = avgtanimg[cy, cx]
    # 6) Isotachen-Stärke:
    eps = 1e-6        
    strength = dist_to_edge[cy, cx]/dist_to_edge[cy, cx].max() + eps
    
    strength = np.clip(strength, 0.0, 1.0).astype(np.float32)
    #-> we want a more equalized field
    strength = strength ** 0.5 # strength is linear by now, probably is sqrt
    # 7) Vektorfeld = Tangente * Stärke
    #    Achtung: Tangenten sind (dy,dx). Flowmap erwartet (0, -y, x, 255).
    vy = avg_tan[:, 0] * strength
    vx = avg_tan[:, 1] * strength

    # 8) Encoding von [-1,1] -> [0..255]
    dt, max = get_datatype(16)
    enc_x, enc_ny = encode_vec_dtype(vx, -vy, 16)  # Reihenfolge: x, -y

    # 9) Flowmap zusammenbauen
    flow = np.zeros((H, W, 4), dtype=dt)
    
    # Kanäle 1/2 nur im Cluster setzen
    flow[cy, cx, 1] = enc_ny
    flow[cy, cx, 2] = enc_x
    flow[cy, cx, 3] = max   #np.sqrt(enc_ny**2 + enc_x**2)
    # Kanal 0 bleibt 0 laut Vorgabe

    return flow



def get_component_mask(mask, point):
    """
    mask : bool-Array (True=Cluster)
    point: (y,x) Koordinate
    return: bool-Array nur mit der Komponente, die den Punkt enthält
    """
    comp, n = label(mask)  # connected components
    y, x = point
    if not mask[y, x]:
        return np.zeros_like(mask, dtype=bool)  # Punkt nicht im Cluster
    cid = comp[y, x]
    return comp == cid

def get_dir_of_points(Startpoint, EndPoint):
    if Startpoint == EndPoint:
        EndPoint = (EndPoint[0] + 1, EndPoint[1])
    v = np.array([EndPoint[0] - Startpoint[0], EndPoint[1] - Startpoint[1]], dtype=float)
    return v / np.linalg.norm(v)

def get_angle_of_dirs(dir1, dir2):
    dot = np.clip(np.dot(dir1, dir2), -1, 1)
    return np.degrees(np.arccos(dot))

def conv_paths_to_skelbin(paths, y, x, flippath = False):
    skel_bin = np.zeros((y, x), np.uint8)
    for path in paths:
        if isinstance(path, (list, tuple, set)):
            if flippath:
                path = np.array([(x, y) for (y, x) in path], dtype=np.int32).reshape((-1, 1, 2))
            else:
                path = np.array(path, dtype=np.int32).reshape((-1,1,2))
            cv2.drawContours(skel_bin, path, -1, 1, 1)
        elif type(path) == np.ndarray:
            skel_bin[~path] = 255
        else:
            raise Exception(f"Well well, what do we have here? an unknown path type: {type(path)}")
    return skel_bin

def conv_path_to_skelbin(path, h, w):
    if isinstance(path, (list, tuple, set)):
        skel_bin = np.zeros((h, w), np.uint8)
        path = np.array(path, dtype=np.int32).reshape((-1,1,2))
        cv2.drawContours(skel_bin, path, -1, 1, 1)
    elif type(path) == np.ndarray:
        skel_bin = (path > 0).astype(np.uint8)
    else:
        raise Exception(f"Well well, what do we have here? an unknown path type: {type(path)}")
    return skel_bin
    
def neighbors(
    y: int, 
    x: int, 
    skel_bin, 
    FilterMode: Literal["NoFilter", "SplitToUnconnected"] = "NoFilter",
    xjunction = None,
    neighmode : Literal["Neigh8", "Neigh4-90"] = "Neigh8"
):            
    neigh = neighmodes[neighmode]
    th, tw = skel_bin.shape
    
    # Alle Nachbarn von (y, x), die im Bild liegen und >0 sind
    nbrs = [(y+dy, x+dx) for dy, dx in neigh
            if 0 <= y+dy < th and 0 <= x+dx < tw and skel_bin[y+dy, x+dx] > 0]
    
    if FilterMode == "NoFilter":
        return nbrs
    
    elif FilterMode == "SplitToUnconnected":
        # Schritt 1: entferne Nachbarn, die mit (y,x) mehr als 1 weiteren Nachbarn teilen
        filtered = []
        removed = []
        filterneigh = neighmodes["Neigh4-90"]
        for n in nbrs:
            ny, nx = n
            # Nachbarn dieses Nachbarn
            n_nbrs = [(ny+dy, nx+dx) for dy, dx in filterneigh
                      if 0 <= ny+dy < th and 0 <= nx+dx < tw and skel_bin[ny+dy, nx+dx] > 0]
            # Schnittmenge mit der Nachbarschaft von (y,x)
            common = set(nbrs) & set(n_nbrs)
            if len(common) <= 1:  # behalte nur, wenn max 1 Nachbar gemeinsam
                filtered.append(n)
            else:
                for c in common:
                    removed.append(c)
        
        # Schritt 2: entferne Paare, die nur 1 Verbindung haben        
        final = []
        for n in filtered:
            if n in removed: continue
            ny, nx = n
            n_nbrs = [(ny+dy, nx+dx) for dy, dx in filterneigh
                      if 0 <= ny+dy < th
                       and 0 <= nx+dx < tw
                       and skel_bin[ny+dy, nx+dx] > 0
                       and not (ny+dy, nx+dy) in removed]
            # Nachbarn dieses Nachbarn innerhalb der gefilterten Liste
            conn = set(n_nbrs) & set(filtered)
            if len(conn) == 0:   # lösche, wenn nur eine Verbindung
                final.append(n)
            else:
                final.append(n)
                for c in conn:
                    removed.append(c)
                    nbrs.remove(c)
        
        if xjunction: xjunction.junctionpointignoredneighbors = removed
        return final

    else:
        raise ValueError(f"Unbekannter FilterMode: {FilterMode}")

def point_to_point_dist(p1, p2):
    return np.sqrt(np.abs(p1[0]-p2[0])**2 + np.abs(p1[1]-p2[1])**2)



#class PathLayers():
#        def __init__(self):
#            class PathLayer():
#                def __init__(self, Layer = 0, Paths : Optional[list[pointpath]] = [], LayerComplete = False):
#                    self.Layer = Layer
#                    self.Paths = Paths
#                    self.LayerComplete = LayerComplete
#                def add_path_to_layer(self, Path : pointpath):
#                    if not Path in self.Paths: 
#                        self.Paths.append(Path)
#            self.Layers : Optional[set[PathLayer]] = {}
#        
#        def get_layer(self, Layer = 0):
#            for lay in self.Layers:
#                if lay.Layer == Layer:
#                    return lay
#                
#        def add_path_to_layer(self, Path : pointpath, Layer = 0):
#            for lay in self.Layers:
#                if lay.Layer == Layer:
#                    lay.add_path_to_layer(Path)
#                    return

#class Serializable:
#    def __init__(self):
#        import uuid
#        self.id = str(uuid.uuid4())
#        self._links = {}
#
#    def __getstate__(self):
#        def proc(x):
#            if isinstance(x, Serializable):
#                return ("__ref__", x.id)
#            if isinstance(x, list):
#                return ("__list__", [proc(y) for y in x])
#            if isinstance(x, tuple):
#                return ("__tuple__", [proc(y) for y in x])
#            if isinstance(x, set):
#                return ("__set__", [proc(y) for y in x])
#            if isinstance(x, dict):
#                return ("__dict__", [[proc(k), proc(v)] for k, v in x.items()])
#            return x
#
#        return {k: proc(v) for k, v in self.__dict__.items()}
#
#    def __setstate__(self, state):
#        self.__dict__.update(state)
#
#    def resolve_links(self, registry):
#        def resolve(x):
#            if isinstance(x, tuple):
#                tag = x[0]
#                if tag == "__ref__":
#                    return registry[x[1]]
#                elif tag == "__list__":
#                    return [resolve(y) for y in x[1]]
#                elif tag == "__tuple__":
#                    return tuple(resolve(y) for y in x[1])
#                elif tag == "__dict__":
#                    return {resolve(k): resolve(v) for k,v in x[1]}
#            elif isinstance(x, list):
#                return [resolve(y) for y in x]
#            elif isinstance(x, dict):
#                return {resolve(k): resolve(v) for k,v in x.items()}
#            return x
#
#        for k,v in list(self.__dict__.items()):
#            if k != "id":
#                self.__dict__[k] = resolve(v)
#
#
#def collect(obj, registry=None, firstlayer=True):
#    if firstlayer:
#        registry = {}
#
#    if isinstance(obj, Serializable):
#        if obj.id in registry:   # schon registriert
#            return registry
#        registry[obj.id] = obj
#        # echte Subobjekte anschauen, nicht Marker
#        vals = obj.__dict__.values()
#    elif isinstance(obj, (list, tuple, set)):
#        vals = obj
#    elif isinstance(obj, dict):
#        vals = list(obj.items())
#    else:
#        return registry
#
#    for v in vals:
#        if isinstance(v, Serializable):
#            collect(v, registry, False)
#        elif isinstance(v, (list, tuple, set)):
#            for x in v:
#                collect(x, registry, False)
#        elif isinstance(v, dict):
#            for k, x in v.items():
#                collect(k, registry, False)
#                collect(x, registry, False)
#
#    if firstlayer:
#        return registry
    
def path_valid(path, ptype : Literal["folder", "file"]):
    if ptype == "file":
        file = os.path.basename(path)
        if len(file) > 3:
            if re.search("\.[a-zA-Z]{1,4}$", file):
                return os.path.isdir(os.path.dirname(path))
    elif ptype == "folder":
        return os.path.isdir(path)
    
class BaseObj():
    def __init__(self, name : str = "BaseObj", savepath : str = ""):
        super().__init__()
        self.name = name
        self.savepath = savepath
    
    def save(self, path : Optional[str] = None):
        if not path:
            if path_valid(self.savepath, "file"):
                save_obj(self, self.savepath)
        else:
             if path_valid(self.savepath, "file"):
                save_obj(self, path)        
        
class pointpath(BaseObj):
    def __init__(self, name = "pointpath", savepath : str = ""):
        super().__init__(name, savepath)
        #upstream, downstream
        self.circlepath = False
        self.junctions : Optional[list[Optional[junction]]] = [None, None]
        #upstream, downstream!!
        self.points = []
        self.pathlength = -1
        self.abspathtraveldistance = -1
        self.downstream = None
        self.networks : List[network] = []
        

    def analyze_path(self):
        self.pathlength = len(self.points)
        self.abspathtraveldistance = (self.points[0][0] - self.points[1][0], self.points[0][1] - self.points[1][1])
        self.set_streamdirection_by_point(self.points[0], True)

    def is_singlepath(self):
        if self.junctions[0] is None and self.junctions[1] is None:
            return True
        return False
    
    def is_junction_to_endpoint(self):
        if (self.junctions[0] is None) != (self.junctions[1] is None):
            return True
        return False

    def flip(self):
        self.junctions = self.junctions[::-1]
        for junc in self.junctions:
            if junc:
                done = False
                mk = None
                for key, ppath in junc.upstreampointandpaths.items():
                    if ppath == self:
                        junc.downstreampointandpaths[key] = ppath
                        done = True
                        mk = key
                        break
                if mk: del junc.upstreampointandpaths[key]
                mk = None
                if not done:
                    for key, ppath in junc.downstreampointandpaths.items():
                        if ppath == self:
                            junc.upstreampointandpaths[key] = ppath  
                            mk = key                          
                            break
                    if mk: del junc.downstreampointandpaths[key]
            if self.circlepath:
                break
        self.points = self.points[::-1]

    def set_streamdirection_by_point(self, Point, IsUpstreamPoint = False):
        if Point in self.points:
            #we reverse the array and junctions according to up/downstream
        
            if IsUpstreamPoint and self.points.index(Point) <= (len(self.points) - 1) / 2:
                self.points = self.points[::-1]
            elif not IsUpstreamPoint and self.points.index(Point) > (len(self.points) - 1) / 2:                
                self.points = self.points[::-1]

            if self.junctions[0]:
                if self.junctions[0].junctionpoint == self.points[0]:
                    return  #we have junctions in right order
                self.junctions = self.junctions[::-1]
                return
            elif self.junctions[1]:
                if self.junctions[1].junctionpoint == self.points[0]:
                    self.junctions =  self.junctions[::-1]
                    return
                return
            else:#endpoint-to-endpoint-path
                return

            raise Exception(f"Something is wrong, cant set_streamdirection_by_point")

            
        else:
            print(f"cant set_streamdirection_by_point for {str(Point)} cuz hes not on the path")

    def set_path_from_junction(self, startjunction : junction, pathdirpoint, network, maxangle = 90):
        self.points = []
        self.junctions[0] = startjunction
        isjunctionpath = startjunction != None
        if isjunctionpath:
            nbs = neighbors(startjunction.junctionpoint[0], startjunction.junctionpoint[1], network.skel_bin)
            if len(nbs) < 3:
                raise Exception("Wtf? This not a junction")
            lastdir = get_dir_of_points(startjunction.junctionpoint, pathdirpoint)
        else:
            nbs = neighbors(pathdirpoint[0], pathdirpoint[1], network.skel_bin)
            lastdir = get_dir_of_points((0,1), (0,0))
        
        #We get the direction from the junction and the pathpoint we want to work on
        closebyendpoint = None
        closebyjunction = None
        currdir = lastdir
        if isjunctionpath:
            prev, curr = startjunction.junctionpoint, get_best_nextpoint_from_dir(lastdir, startjunction.junctionpoint, nbs, maxangle)
        else:
            prev, curr = pathdirpoint, get_best_nextpoint_from_dir(lastdir,pathdirpoint, nbs, maxangle)
        keepRunning = True
        self.points = [prev, curr]
        traveldist = 0
        while keepRunning:
            nbs = neighbors(curr[0], curr[1], network.skel_bin)
            nbs = [nb for nb in nbs if not nb in self.points]
            if len(nbs) == 1:
                nbs = nbs[0]
            elif len(nbs) > 1:
                nbs = get_best_nextpoint_from_dir(lastdir, curr, nbs, maxangle)
                if not nbs:
                    if not curr in network.endpoints: network.endpoints.append(curr)
                    if not curr in network.doneendpoints: network.doneendpoints.append(curr)
                    break
            elif len(nbs) == 0:
                if not curr in network.endpoints: network.endpoints.append(curr)
                if not curr in network.doneendpoints: network.doneendpoints.append(curr)
                break
                raise Exception("No more neighbours? We've come abroad!! There should be an endpoint!")
            traveldist += 1
            prev, curr = curr, nbs
            lastdir = currdir
            currdir = get_dir_of_points(prev, curr)
            if get_angle_of_dirs(currdir, lastdir) > maxangle:                
                if not prev in network.endpoints: network.endpoints.append(prev)
                if not prev in network.doneendpoints: network.doneendpoints.append(prev)
                break

            self.points.append(nbs)
            #first we check if there is any junction nearby. if so, we take it as target.
            if isjunctionpath:
                if not closebyjunction: 
                    closebyjunction, dist = network.is_point_close_to_junction(nbs, 3)
                    if closebyjunction == startjunction and traveldist < 4: 
                        closebyjunction = None
                    otherjunctionpathpoints = {p : "u" for p in startjunction.upstreampointandpaths.keys() if not p == pathdirpoint} | {p : "d" for p in startjunction.downstreampointandpaths.keys() if not p == pathdirpoint}
                    if curr in otherjunctionpathpoints.keys(): #elif closebyjunction == startjunction and traveldist >= 4:
                        #this means we have gone in circles and should add the path leading the other way around to "self"
                        #since the junction has 2 outputs but it is only 1 path! 
                        # -> we ignore the other point in the next outer loop if it already has a path
                        self.circlepath = True
                        self.junctions[1] = startjunction
                        if otherjunctionpathpoints[curr] == "u":
                            startjunction.upstreampointandpaths[curr] = self
                        else:                            
                            startjunction.downstreampointandpaths[curr] = self
                else:
                    dist = point_to_point_dist(closebyjunction.junctionpoint, curr)

                if closebyjunction: #this is to steer towards the next junction point
                    if dist == 0:
                        self.junctions[1] = closebyjunction
                        break
                    lastdir = get_dir_of_points(curr, closebyjunction.junctionpoint)
                    continue
            #if we have no junction closeby, we search for closeby endpoints. if we have an endpoint, this is the target
            if not closebyendpoint: 
                closebyendpoint, dist = network.is_point_close_to_endpoint(nbs, 3)
                if closebyendpoint == pathdirpoint: closebyendpoint = None
            else:
                dist = point_to_point_dist(pathdirpoint, curr)

            if closebyendpoint: #this is to steer towards the endpoint
                if dist == 0:
                    network.doneendpoints.append(nbs)
                    break
                lastdir = get_dir_of_points(curr, closebyendpoint)            
        self.analyze_path()

    def remove_pointpath(self):
        for j in self.junctions:
            if j:
                broke = False
                for p, pp in j.upstreampointandpaths.items():
                    if pp == self:
                        del j.upstreampointandpaths[p]
                        broke = True
                        break
                if not broke:
                    for p, pp in j.downstreampointandpaths.items():
                        if pp == self:
                            del j.downstreampointandpaths[p]
                            broke = True
                            break
        for n in self.networks:
            if n:
                for pp in n.singlepaths:
                    if pp == self:
                        n.singlepaths.remove(pp)
                        return
                    
    def collect_all(self, out):
        if not self in out["pointpath"]: out["pointpath"].append(self)
        return out
    
class junction(BaseObj):    
    def __init__(self, junctionpoint, networklist = [], name = "junction", savepath : str = ""):
        super().__init__(name, savepath)
        self.junctionpoint = junctionpoint
        self.upstreampointandpaths      : Dict[Tuple[int,int], pointpath] = {}
        self.downstreampointandpaths    : Dict[Tuple[int,int], pointpath] = {}
        self.junctionpointignoredneighbors = []
        if not isinstance(networklist, list):
            raise Exception(f"No valid networklisttype!! {type(networklist)}")
        for net in networklist:
            if not isinstance(net, network):
                raise Exception(f"No valid networktype!! {type(net)}")
        self.networks : List[network] = networklist
        self.pointsanglecount = {}
        self.maindownstream = None
        self.currentflowdir = (0.0, 0.0)
    
    def init_junction(self, initnetwork, networkfullyinitialized = False, debugprint = False):
        if not self.gen_currentvecs(2, initnetwork.skel_bin, debugprint):
            if debugprint: print(f" didnt take Junction point: {str(self.junctionpoint[::-1])} ->")   
            return False
        self.gen_pointsanglecount()
        if not self.clean_with_pointsanglecount():
            return False
        self.analyze_pointsanglecount()
        if networkfullyinitialized:
            self.gen_pointpaths(initnetwork)
        return True
    
    def get_all_paths(self) -> List[pointpath]:
        pointpaths = []
        for pp in list(self.upstreampointandpaths.values()) + list(self.downstreampointandpaths.values()):
            if pp: 
                if not pp in pointpaths: pointpaths.append(pp)
        return pointpaths
    
    def get_path_from_point(self, Point : Tuple[int,int]) -> pointpath | None:
        for path in (self.upstreampointandpaths | self.downstreampointandpaths).values():
            if path:
                if Point in path.points:
                    return path
        return None

    def get_all_connected_juncs(self, connectedlist = [], Level = 0):
        if Level == 0:
            connectedlist = []
        if not self in connectedlist: connectedlist.append(self)
        workpathdic = self.upstreampointandpaths | self.downstreampointandpaths
        for juncpath in workpathdic.values():
            for junc in juncpath.junctions:             #-> circle path
                if junc and not junc in connectedlist and not junc == self:
                    junc.get_all_connected_juncs(connectedlist, Level + 1)
        if Level != 0:
            return connectedlist
        return [junc for junc in connectedlist]

    def add_to_network(self, addnetwork):
        if type(addnetwork) == network:
            if addnetwork in self.networks:
                self.networks.append(addnetwork)

    def flip(self):
        allpaths = set([x for x in self.upstreampointandpaths.values()]) | set([x for x in self.downstreampointandpaths.values()])
        donepaths = []
        for ppath in allpaths:
            if not ppath in donepaths:
                ppath.flip()
                donepaths.append(ppath)
    
    def gen_pointpaths(self, alternetwork = None):
        worknetwork = self.networks[0]
        if alternetwork and type(alternetwork) == network: worknetwork = alternetwork
        j = 0
        for upstrpt in self.upstreampointandpaths.keys():
            j += 1
            if self.upstreampointandpaths[upstrpt] is None:
                ppath = pointpath("Pointpath_" + str(j))
                ppath.set_path_from_junction(self, upstrpt, worknetwork)
            else:
                continue #-> it means we have a path that is going in circles
            for k, pjunc in enumerate(ppath.junctions):
                if pjunc: # and not pjunc == self: -> pjunc can be self since we can go in circles, but then again we skip it as whole because we set everything when finding this out
                    #if we found another junction while creating the path
                    #-> we register the path in the other junctions dictionary unless
                    #   the other junction already has a path for this, then we take this path as ours
                    #   and link ourself into the paths junctions
                    otherjuncspath = pjunc.get_path_from_point(ppath.points[int(len(ppath.points)/2)])
                    if otherjuncspath:
                        if not self in otherjuncspath.junctions:
                            for i, junc in enumerate(otherjuncspath.junctions):
                                if not junc == pjunc:
                                    otherjuncspath.junctions[i] = junc
                                    break
                        ppath = otherjuncspath
                    break
            self.upstreampointandpaths[upstrpt] = ppath
        j = 0
        for dwstrpt in self.downstreampointandpaths.keys():
            j += 1
            if self.downstreampointandpaths[dwstrpt] is None:
                ppath = pointpath("Pointpath_" + str(j))
                ppath.set_path_from_junction(self, dwstrpt, worknetwork)
            else:
                continue
            for k, pjunc in enumerate(ppath.junctions):
                if pjunc: # and not pjunc == self:
                    otherjuncspath = pjunc.get_path_from_point(ppath.points[int(len(ppath.points)/2)])
                    if otherjuncspath:
                        if not self in otherjuncspath.junctions: #if we are already registered, everything good
                            for i, junc in enumerate(otherjuncspath.junctions):
                                if not junc == pjunc:
                                    otherjuncspath.junctions[i] = junc
                                    break
                        ppath = otherjuncspath
                    break
            self.downstreampointandpaths[dwstrpt] = ppath

    def delpoints(self, elimpoints):        
        for elimpoint in elimpoints:    #we delete points from the list
            self.pointsanglecount.pop(elimpoint)
        for point in self.pointsanglecount.keys():   #we delete points from the groups
            for group, values in self.pointsanglecount[point].items():
                self.pointsanglecount[point][group] = [val for val in values if not val[0] in elimpoints]
    #not done yet, need for dynamic shit, but why need dynamic shit?

    def gen_currentvecs(self, step_out, otherskelbin, debugprint = False, strictstepout = True):
        #we generate the points and angles to the points for each other point of the junction relative to the junction
        if debugprint: print(f"Junction point: {str(self.junctionpoint[::-1])}, Start") 
        currentneighbours = neighbors(*self.junctionpoint, otherskelbin, FilterMode= 'SplitToUnconnected', xjunction=self)
        if len (currentneighbours) < 3:
            return None      
        if debugprint: print(f"Junction point: {str(self.junctionpoint[::-1])}, neighs: [{str([n[::-1] for n in currentneighbours])}]")      
        vecs = {}
        ignorepoints = [p for p in currentneighbours]
        ignorepoints.append(self.junctionpoint)
        y, x = self.junctionpoint[0], self.junctionpoint[1]
        hasupordownstreampaths = len(self.upstreampointandpaths | self.downstreampointandpaths) == 0

        oldyx = None
        upstr = False
        for (ny, nx) in currentneighbours:
            # Arm um step_out Pixel hinaus verfolgen   
            prev, curr = (y, x), (ny, nx)
            steps = 1
            prevdir = get_dir_of_points(prev, curr)
            currdir = prevdir
            oldyx = None
            HitJunc = None
            while steps < step_out:
                newcurrneighbours = neighbors(*curr, otherskelbin)                        
                further = [p for p in newcurrneighbours if p != prev and not p in ignorepoints]
                if len(further) > 1:
                    further = [get_best_nextpoint_from_dir(prevdir, (y, x), further, 90)]  # Multiple choices-> choose so that direction changes least to prev or start dir
                elif len(further) == 0: #endpoint reached                    
                    break
                        
                #Getting up/downstream and its index
                if further[0] in self.upstreampointandpaths.keys():
                    upstr = True
                    oldyx = further[0]
                elif further[0] in self.downstreampointandpaths.keys():
                    upstr = False
                    oldyx = further[0]

                ignorepoints.append(further[0])
                prev, curr = curr, further[0]
                if not HitJunc:
                    HitJunc, Distance = self.networks[0].is_point_close_to_junction(curr, 2)
                    if HitJunc == self:
                        HitJunc = None
                    elif HitJunc:   #if we have another closeby junction, we stop our search for points
                        break
                prevdir = currdir
                currdir = get_dir_of_points(prev, curr)
                steps += 1
            if oldyx and curr != oldyx:
                if upstr:
                    self.upstreampointandpaths[curr] = self.upstreampointandpaths.pop(oldyx)                    
                else:
                    self.downstreampointandpaths[curr] = self.downstreampointandpaths.pop(oldyx)
                    
            if steps != step_out and strictstepout:
                continue
            # Richtung aus Junction zum "step_out"-Punkt
            v = np.array([curr[0] -y, curr[1] - x], dtype=float)
            if np.linalg.norm(v) < 1e-6:
                continue
            v /= np.linalg.norm(v)           
            vecs[curr] = v

        if len(vecs) > 2:
            self.vecs = vecs
            return vecs
        self.vecs = None
        return None
    
    def gen_pointsanglecount(self):
        # "Arme" clustern: zwei Richtungen zusammen, wenn < x° Unterschied
        #we generate a set of points and how often they are in a certain angle to other streams
        usedind = {}
        uniqueinds = []
        arms = 0
        directionsofpoints = [val for val in self.vecs.values()]
        pointsleadingaway = [vec for vec in self.vecs.keys()]
        self.pointsanglecount = {point : {"less20" : [], "less90" : [], "greatereq90" : []} for point in pointsleadingaway}
        for i in range(len(directionsofpoints)):
            if not i in usedind.keys():
                usedind[i] = []
            arms += 1
            for j in range(len(directionsofpoints)):
                if i == j: #used[j]:
                    continue
                if j in usedind.keys():
                    if i in usedind[j]:
                        continue
                    usedind[j].append(i)
                else:
                    usedind[j] = [i]
                dot = np.clip(np.dot(directionsofpoints[i], directionsofpoints[j]), -1, 1)
                angle = np.degrees(np.arccos(dot))
                if angle < 20:  # nahezu gleiche Richtung → gleicher Arm
                    self.pointsanglecount[pointsleadingaway[i]]["less20"].append([pointsleadingaway[j], angle])
                elif angle < 90: #flows parallel
                    self.pointsanglecount[pointsleadingaway[i]]["less90"].append([pointsleadingaway[j], angle])
                elif angle >= 90: #flows in same direction                  
                    self.pointsanglecount[pointsleadingaway[i]]["greatereq90"].append([pointsleadingaway[j], angle])
            uniqueinds.append(i)
    
    def clean_with_pointsanglecount(self):
        # clean the points for the less20's.
        # -> if they have 2 partners, they eliminate the other point.
        # -> loop until no points with partners
        # -> if we dont only have more than 3 partners left, we choose one randomly and eliminate the others
        No2Partners = 1
        savedpoints = []
        elimpoints = []
        pointsanglekeys = self.pointsanglecount.keys()
        while No2Partners and not len(savedpoints) + len(elimpoints) == len(self.pointsanglecount):            
            for point in pointsanglekeys:
                if not (point in savedpoints or point in elimpoints):
                    if len(self.pointsanglecount[point]["less20"]) == 0:
                        if not point in savedpoints: savedpoints.append(point)
                    elif len(self.pointsanglecount[point]["less20"]) == 1:
                        No2Partners = -1
                        if not point in savedpoints: savedpoints.append(point)
                        for p in self.pointsanglecount[point]["less20"]:
                            if not p in elimpoints: elimpoints.append(p[0])
                    elif len(self.pointsanglecount[point]["less20"]) > 1:
                        if No2Partners == 0: #We ignore those better connected until we dont have any single 
                            for p in self.pointsanglecount[point]["less20"]:
                                if not p in elimpoints: elimpoints.append(p[0])
                            break
            if No2Partners == 1: #-> 1 means nothing changed
                No2Partners = 0  #-> 0 means change next round
            else: #if -1 or 0, we changed something, and reset to initial state
                No2Partners = 1

        #clean up
        if len(self.pointsanglecount) - len(elimpoints) <=2:   #We reduced the paths to less than two, so it's not a junction at all
            return False        
        self.delpoints(elimpoints)
        return True
        
    def analyze_pointsanglecount(self, altermaindownstream = None):
        #depending on which path has the most big angles to the other paths thats considered downstream for now
        #this is to seperate the up/downstream for each junction
        #this allows us to later only flip the up/downstream group when making a coherent network
        if len(self.downstreampointandpaths) == 0: self.downstreampointandpaths = {}
        if len(self.upstreampointandpaths) == 0: self.upstreampointandpaths = {}
        maindownstream = None
        if altermaindownstream:
            maindownstream = altermaindownstream
        else:
            maindownstream = None
            for point in self.pointsanglecount.keys():
                if maindownstream is None:  #initializing
                    maindownstream = point
                    continue
                if len(self.pointsanglecount[point]["greatereq90"]) >  len(self.pointsanglecount[maindownstream]["greatereq90"]):
                    maindownstream = point
        if maindownstream is None:
            raise Exception("BLYAT!")
        self.maindownstream = maindownstream
        self.currentflowdir = get_dir_of_points(maindownstream, self.junctionpoint)
        upanddowndict = self.upstreampointandpaths | self.downstreampointandpaths
        firstinit = len(upanddowndict.keys()) == 0
        downstreampointandpaths = {}
        upstreampointandpaths = {}
        #up and down dicts get managed beforehand. old paths will be taken to the new points for the dictionaries
        if not firstinit:
            if maindownstream in upanddowndict.keys():
                downstreampointandpaths[maindownstream] = upanddowndict[maindownstream]
            else:
                downstreampointandpaths[maindownstream] = self.networks[0].get_path_from_endpoint(maindownstream)
        else:
            downstreampointandpaths[maindownstream] = None #for later use when we get the actual paths
        #We loop through the points less90° to the "downest" stream, those are upstream parallel -> 
        for point in self.pointsanglecount[maindownstream]["less90"]:
            if not firstinit:
                if point[0] in upanddowndict.keys():
                    downstreampointandpaths[point[0]] = upanddowndict[point[0]]
                else:
                    downstreampointandpaths[point[0]] = self.networks[0].get_path_from_endpoint(point)
            else:
                downstreampointandpaths[point[0]] = None
        #And then we loop through the points greater90° to the "downest" stream, those are upstream parallel
        for point in self.pointsanglecount[maindownstream]["greatereq90"]:
            if not firstinit:   #this not worksnt!!
                if point[0] in upanddowndict.keys():
                    upstreampointandpaths[point[0]] = upanddowndict[point[0]]
                else:
                    upstreampointandpaths[point[0]] = self.networks[0].get_path_from_endpoint(point)
            else:
                upstreampointandpaths[point[0]] = None
        self.downstreampointandpaths = downstreampointandpaths
        self.upstreampointandpaths = upstreampointandpaths

    def collect_all(self, out):
        if not self in out["junction"]: out["junction"].append(self)
        for pp in self.upstreampointandpaths.values():
            pp.collect_all(out)
        for pp in self.downstreampointandpaths.values():
            pp.collect_all(out)
        return out
    
class network(BaseObj):
    def __init__(self, clustermask = None, SkeletonType : Literal["lee", "zhang", "weird_splitted_skeleton"] = "weird_splitted_skeleton", name = "network", savepath : str = ""):
        global file
        super().__init__(name, savepath)
        if isinstance(clustermask, np.ndarray):
            self.clustermask = clustermask
            if SkeletonType in ["lee", "zhang"]:
                self.skeleton = morphology.skeletonize(clustermask, method=SkeletonType)
            elif SkeletonType == "weird_splitted_skeleton":
                self.skeleton = weird_splitted_skeleton(clustermask, ShowSteps=False, masksmoothmode=["Gaussian"])
            if AccessMngr.can_access(["networkclass", "networks", "skeleton"]): FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, self.skeleton, "_" + output_skeletonadder + f"_", "skeleton", AddAlpha=True, Normalize=True)
        else:
            self.clustermask = None
            self.skeleton = None

        self.parent                 : cluster = None
        self.network                : Dict[junction, None] = {}
        self.junctionpointlist      : Optional[list[(int,int)]] = []  #
        self.singlepaths            : Optional[list[pointpath]] = [] #
        self.skel_bin = None
        self.poppinglist = False
        self.doneendpoints = []
        self.flowmapobjs            : List[flowmapobj] = []
        #we only initialize network if we have a skeleton to work on
        #else it will be an empty network to be filled elsewise
        self.endpoints = []
        if type(self.skeleton) != type(None):    
            self.gen_endpoints_and_junctions()
            if AccessMngr.can_access(["networkclass", "networks", "junctions"]): SaveFlowmap(outputfolder, file.name, file.format, PointsToFlowmap(self.junctionpointlist, currh, currw, 1, radius = 0), f"_{output_mergeresolve}_aftergenendpAjunc_", "innetwork_init_junclist")

    #PatLayers = PathLayers()

    def is_point_close_to_junction(self, point, tolerance = 1):
        for junc in self.network:
            dist = point_to_point_dist(junc.junctionpoint, point)
            if dist <= tolerance:
                return junc, dist
        return None, -1
    
    def is_point_close_to_endpoint(self, point, tolerance = 1):
        for endp in self.endpoints:
            dist = point_to_point_dist(endp, point)
            if dist <= tolerance:
                return endp, dist
        return None, -1 
    
    def get_network_pointpaths(self) -> List[pointpath]:
        pointpaths = []
        if len(self.network) > 0:
            for j in self.network.keys():
                for pp in list(j.upstreampointandpaths.values()) + list(j.downstreampointandpaths.values()):
                    if not pp in pointpaths: pointpaths.append(pp)
        else:
            pointpaths.extend(self.singlepaths)
        return pointpaths
    
    def add_points_as_junctions(self, pointlist = None):
        added = False
        i = 0
        if type(pointlist) == list or type(pointlist) == set:            
            for p in pointlist:
                if type(p) == list or type(p) == set or type(p) == tuple:
                    if len(p) == 2:
                        i += 1
                        junc = junction(p, [self], "Junction_" + str(i))
                        if junc.init_junction(self):
                            self.append(junc)
                            added = True
        #we should be good because of strikt filtering rules when initializing the junction
        #self = cluster_points(self, self.skel_bin)
        if added:
            self.gen_junctionpointlist()
            if AccessMngr.can_access(["networkclass", "networks", "junctions"]): SaveFlowmap(outputfolder, file.name, file.format, PointsToFlowmap(self.junctionpointlist, currh, currw, 1, radius = 0), f"_{output_mergeresolve}_junctionpointlist_afteraddptasjunc_", "innetwork_junctionpointlist")
            

    def append(self, junction : junction):
        self.network[junction] = None

    def poplist(self, topoplist):
        if type(topoplist) == list:
            topoplist.sort()
            topoplist = topoplist[::-1]
            self.poppinglist = True
            for topop in topoplist:
                self.pop(topop, True)
            self.poppinglist = False
            self.gen_junctionpointlist()

    def get_junction_from_endpoint(self, Point) -> junction | None:
        for junc in self.network:
            endp = junc.get_path_from_point(Point)
            if not endp is None:
                return junc
        return None
    
    def get_path_from_endpoint(self, Point) -> pointpath | None: 
        for junc in self.network:
            endp = junc.get_path_from_point(Point)
            if not endp is None:
                return endp
        for singp in self.singlepaths:
            if Point in singp.points:
                return singp
        return None
    
    def pop(self, i, poplist = False):
        if type(i) == int:
            self.network.pop([k for k in self.network.keys()][i])
        if isinstance(i, junction):
            self.network.pop(i)
        if not self.poppinglist: self.gen_junctionpointlist()
    
    def gen_endpoints_and_junctions(self, newskel = None):
        global currh, currw
        if newskel:
            self.skeleton = newskel
        workskel = self.skeleton        
        # Falls Konturen gegeben → zu binärem Skeleton-Bild machen
        self.skel_bin = conv_path_to_skelbin(workskel, self.skeleton.shape[0], self.skeleton.shape[1])
        #if isinstance(workskel, (list, tuple)):
        #    self.skel_bin = np.zeros((currh, currw), np.uint8)
        #    cv2.drawContours(self.skel_bin, workskel, -1, 1, 1)
        #else:    
        #    self.skel_bin = (workskel > 0).astype(np.uint8)
        
        # Pixel-Koordinaten
        coords = np.column_stack(np.nonzero(self.skel_bin))
        
        if AccessMngr.can_access(["networkclass", "networks", "skeleton"]): SaveFlowmap(outputfolder, file.name, file.format, self.skel_bin, f"_{output_mergeresolve}_skelbin_", "innetwork_skelbin", True, True)

        # Knotentypen finden
        # degree berechnen
        degree = {(y,x): len(neighbors(y,x, self.skel_bin)) for y,x in coords}
        endpoints = {p for p, d in degree.items() if d == 1}
        junctions = {p for p, d in degree.items() if d > 2}
        self.endpoints = [p for p in cluster_points(endpoints, self.skel_bin)]
        if AccessMngr.can_access(["networkclass", "networks", "endpoints"]): SaveFlowmap(outputfolder, file.name, file.format, PointsToFlowmap(endpoints, currh, currw, 1, None, 0), f"_endpoints_", "innetwork_endpoints")
        if AccessMngr.can_access(["networkclass", "networks", "juncions"]): SaveFlowmap(outputfolder, file.name, file.format, PointsToFlowmap(junctions, currh, currw, 1, None, 0), f"_rawjunctions_", "innetwork_endpoints")
        self.add_points_as_junctions(junctions)

    def gen_junctionpointlist(self):
        self.junctionpointlist = [junc.junctionpoint for junc in self.network]

    def rename_network_members_numeration(self, networks):
        #->implementation for working inside of clusters. we want to build names based on network relations for ID
        if isinstance(networks, network):
            networks = [networks]
        elif not isinstance(networks, list):
            raise Exception(f"Error in 'rename_network_members_numeration': no valid networks type ({type(networks)})")
        
        
        for i, net in enumerate(networks):
            net.layername = f"Network_{i}"
            net.name = f"Network_{i}"
            pointpathcount = 1
            for j, junc in enumerate(net.network.keys()):
                junc.layername = "_".join([net.layername, f"Junction_{j}"])
                junc.name = f"Junction_{j}"
                for k, ppath in enumerate((junc.upstreampointandpaths | junc.downstreampointandpaths).values()):
                    ppath.layername = "_".join([junc.layername, f"Junctionpath_{pointpathcount}"])
                    ppath.name = f"Junctionpath_{pointpathcount}"
                    pointpathcount += 1
            for k, ppath in enumerate(net.singlepaths):
                ppath.layername = "_".join([net.layername, f"Singlepath_{pointpathcount}"])
                ppath.name = f"Singlepath_{pointpathcount}"
                pointpathcount += 1

    def connect_split_process_network_to_networks(self):
        #We loop through junctions and generate pointpaths which lead to other junctions or nothing (ends)
        self.doneendpoints = []
        rem = []
        for junc in self.network:
            if not isinstance(junc, junction):
                rem.append(junc)
        for i in rem[::-1]:
            del self.network[i]
        for i, junc in enumerate(self.network):            
            junc.gen_pointpaths()
            junc.name = "Junction_" + str(i)
        #we loop through endpoints to check if all are taken. 
        # if they are not taken, it is because it's an endpoint-to-endpoint-connection
        
        self.singlepaths = []
        while len(self.doneendpoints) < len(self.endpoints):
            lastendpointcount = len(self.endpoints)
            i = 0
            for endpoint in self.endpoints:
                if not endpoint in self.doneendpoints:
                    i += 1
                    self.doneendpoints.append(endpoint)
                    singlepath = pointpath()
                    singlepath.set_path_from_junction(None, endpoint, self, 360)
                    singlepath.name = "Pointpath_" + str(i)
                    self.singlepaths.append(singlepath)
                    if lastendpointcount != len(self.endpoints): break
                    
        #-> we now have linked every class-obj of the network
        #we split the network into a list of networks that are actually connected

        newnetworks = self.split_network_by_connections()
        ##DEBUGGING
        ##remove later
        rpfilename = file.name.replace(file.format,"")
        if rpfilename == "phoenix_all_multjunc_3":
            FlowTargetPoint = "RightDown"
        else:
            FlowTargetPoint = "Middle"
        ##

        #now we sort the network so the flow direction of each subpath is along the longest path 
        for newnet in newnetworks:
            newnet.sort_network(FlowTargetPoint=FlowTargetPoint)
        self.rename_network_members_numeration(newnetworks)
        return newnetworks

    def gen_endpoints_from_junctionpaths(self):
        self.endpoints = []
        for junc in self.network.keys():
            juncpaths = junc.upstreampointandpaths | junc.downstreampointandpaths
            for juncpath in juncpaths.values():
                if juncpath.junctions[0] is None or juncpath.junctions[1] is None:
                    if juncpath.points[0] == junc.junctionpoint:
                        self.endpoints.append(juncpath.points[-1])
                    elif juncpath.points[-1] == junc.junctionpoint:
                        self.endpoints.append(juncpath.points[0])

    def gen_skel_and_skelbin_from_junctionpaths(self, y, x):
        self.skel_bin = np.zeros((y, x), np.uint8)
        juncpaths = [[p[::-1] for p in pat.points] for junc in self.network.keys() for pat in (junc.upstreampointandpaths | junc.downstreampointandpaths).values()]
        self.skel_bin = None
        self.skeleton = None
        if len(juncpaths) > 0:
            self.skel_bin = conv_paths_to_skelbin(juncpaths, y, x)
            self.skeleton = self.skel_bin.astype(bool)
            for junc in self.network:
                junc.skel_bin = self.skel_bin
            

    def split_network_by_connections(self):
        #splitting the network into networks of linked class-objs
        #starting with singlepaths, those are their own networks
        singlepathnetworks = [network() for singpath in self.singlepaths]
        for i, singpathnet in enumerate(singlepathnetworks):
            singpathnet.skel_bin = conv_path_to_skelbin(self.singlepaths[i].points, self.skel_bin.shape[0], self.skel_bin.shape[1])
            singpathnet.singlepaths.append(self.singlepaths[i])
            singpathnet.name = "Network_" + str(i)
            for j in [1,-1]:
                singpathnet.endpoints.append(self.singlepaths[i].points[j])
            singpathnet.skeleton = singpathnet.skel_bin.astype(bool)
            singpathnet.clustermask = get_component_mask(self.clustermask, singpathnet.singlepaths[0].points[0])
        #now we seperate by junctions that arent connected to each other
        junctionnetworks = []
        searchjuncs = [junc for junc in self.network]
        #while len(searchjuncs) != 0:
        i = 0
        while 0 < len(searchjuncs):
            i += 1
            junc = searchjuncs.pop(0)
            connectedjuncs = junc.get_all_connected_juncs()
            newnet = network()
            newnet.name = "Network_" + str(i)
            newnet.network = {junc : None for junc in connectedjuncs}
            newnet.clustermask = get_component_mask(self.clustermask, [tnet for tnet in newnet.network.keys()][0].junctionpoint)
            newnet.gen_endpoints_from_junctionpaths()
            if AccessMngr.can_access(["networkclass", "networks", "cluster"]): SaveFlowmap(outputfolder, file.name, file.format, newnet.clustermask, output_clusteradder + "_test")
            newnet.gen_skel_and_skelbin_from_junctionpaths(*self.skeleton.shape)
            newnet.gen_junctionpointlist()
            junctionnetworks.append(newnet)
            junc.networks.insert(0, newnet)
            
            for i in range(1, len(connectedjuncs)): 
                if connectedjuncs[i] in searchjuncs: 
                    searchjuncs.remove(connectedjuncs[i])
                if not newnet in connectedjuncs[i].networks: 
                    connectedjuncs[i].networks.insert(0, newnet)
        return singlepathnetworks + junctionnetworks

    def get_FlowTargetPoint(self, FlowTargetPoint):
        #get the flow target point from string. flow target decides which endpoints are "downstream"
        match FlowTargetPoint:
            case "Middle":
                FlowTargetPoint = (int(self.skeleton.shape[0] / 2), int(self.skeleton.shape[1] / 2))
            case "LeftUp":
                FlowTargetPoint = (0,0)
            case "LeftDown":
                FlowTargetPoint = self.skeleton.shape
                FlowTargetPoint = (FlowTargetPoint[0], 0)                    
            case "RightUp":
                FlowTargetPoint = self.skeleton.shape
                FlowTargetPoint = (0, FlowTargetPoint[0])
            case "RightDown":
                FlowTargetPoint = self.skeleton.shape
        return FlowTargetPoint
    
    def sort_network(self,
                     FlowTargetPoint : Literal["LeftUp", "RightUp", "LeftDown", "RightDown", "Middle"] | Tuple[int,int] = "Middle",
                     SortMode : Literal["FurthestEndpoints"] = "FurthestEndpoints"):
        
        if isinstance(FlowTargetPoint, str):
            FlowTargetPoint = self.get_FlowTargetPoint(FlowTargetPoint)
                    
        match SortMode:
            case "FurthestEndpoints":
                FurthestPoints = self.get_furthest_endpoints()
                if not FurthestPoints:
                    return #raise Exception("ey was isch hier los gibt kei punkte oda was")
                #We get starting point by choosing which one is closer to our flow target point
                if point_to_point_dist(FurthestPoints[0], FlowTargetPoint) < point_to_point_dist(FurthestPoints[1], FlowTargetPoint):
                    StartPoint = FurthestPoints[0]
                else:
                    StartPoint = FurthestPoints[1]
                #The Startpoint from which the flow will start
                #we now follow the stream and arrange upstreams and downstreams
                CurrPath = self.get_path_from_endpoint(StartPoint)
                CurrPath.set_streamdirection_by_point(StartPoint, True)
                CurrPathsJuncs = [junc for junc in CurrPath.junctions if not junc is None]
                if len(CurrPathsJuncs) == 0:
                    return  #It's singlepath
                donepaths = [CurrPath]
                Donejuncs = []
                #-> there's the chance of there only beeing a path from singlepathpoints without junction
                
                CurrJunc : junction = CurrPathsJuncs[0]
                #PatLayers = PathLayers()               
                #self.walk_network_parallel_recursive(CurrJunc=CurrJunc, DonePaths=donepaths, DownStreamSearchPath=CurrPath, Layer = 0)
                self.walk_network_by_unused_paths_recursively(CurrJunc=CurrJunc, DoneJuncs=Donejuncs, DonePaths=donepaths, DownOrUpStreamSearchPath=CurrPath)

    def is_in_donepathsdict(path : pointpath, donepathsdict : Optional[dict[int, Optional[list[pointpath]]]]):
        for key, val in donepathsdict.items():
            if path in val:
                return key
        return False
    def add_to_donepathsdict(path : pointpath, donepathsdict : Optional[dict[int, Optional[list[pointpath]]]], Layer : int = 0):
        if not Layer in donepathsdict.keys():
            donepathsdict[Layer] = [path]
        if not path in donepathsdict[Layer] : donepathsdict[Layer].append(path)
        return donepathsdict
    
    def walk_network_by_unused_paths_recursively(self, CurrJunc : junction, DoneJuncs : List[junction], DonePaths : List[pointpath], DownOrUpStreamSearchPath : pointpath = None, IsDownstream : bool = True):
        #walks the network, starting from "downstream"->pointpath and its junction
        #flips the up and downstreams of the current junction according to an input path
        #recursively calls itself through the junctions connected to the given junction, setting those flowdirs
        #ignores taken paths
        #-> this assumes the angles of the junctions are properly
        #-> this should be ensured by taking points that are some distance away from the junction to generate
        #   better angles (the input angle 1 pixel away would otherwise always be in 45° steps)
        #TODO : prioritize "straight" flow - paths that are straight are of the main flow
        #
        DoneJuncs.append(CurrJunc)
        if DownOrUpStreamSearchPath:
            if not CurrJunc.gen_currentvecs(30, self.skel_bin, strictstepout=False):
                raise Exception(f"Cant generate currentvecs for currjunc")                
            CurrJunc.gen_pointsanglecount()
            CurrJunc.analyze_pointsanglecount()
            #We flip the junction depending on if the provided path is in the down or upstream dictionary 
            if IsDownstream and (not DownOrUpStreamSearchPath in CurrJunc.downstreampointandpaths.values()):
                CurrJunc.flip()
            elif not IsDownstream and (not DownOrUpStreamSearchPath in CurrJunc.upstreampointandpaths.values()):
                CurrJunc.flip()
        ppaths = [x for x in CurrJunc.upstreampointandpaths.values()]
        for path in ppaths:
            if not path in DonePaths:
                DonePaths.append(path)
                path.set_streamdirection_by_point(CurrJunc.junctionpoint, True)
                for pjunc in path.junctions:
                    if pjunc and not pjunc in DoneJuncs:
                        self.walk_network_by_unused_paths_recursively(pjunc, DoneJuncs=DoneJuncs, DonePaths=DonePaths, DownOrUpStreamSearchPath=path, IsDownstream=True)
                        break
        ppaths = [x for x in CurrJunc.downstreampointandpaths.values()]
        for path in ppaths:
            if path:
                if not path in DonePaths:
                    DonePaths.append(path)
                    path.set_streamdirection_by_point(CurrJunc.junctionpoint, False)
                    for pjunc in path.junctions:
                        if pjunc and not pjunc in DoneJuncs:
                            self.walk_network_by_unused_paths_recursively(pjunc, DoneJuncs=DoneJuncs, DonePaths=DonePaths, DownOrUpStreamSearchPath=path, IsDownstream=False)
                            break

    def walk_network_parallel_recursive(self, CurrJunc : junction, DonePaths : Optional[dict[int, Optional[list[pointpath]]]], DownStreamSearchPath : pointpath = None, Layer = 0, InternalRunTime = 0):
        #walks the network by setting down and upstream to each path of a junction 
        # relative to incomming paths from downstream 
        #before stepping into the junctions of those paths and repeating the process layer by layer
        # IS NOT FINISHED - We take a simplified approach that just walks the net from the start
        # to each untaken path, trusting the angles of the junctions will be correct for each junction 
        
        if DownStreamSearchPath:
            CurrJunc.gen_currentvecs(30, self.skel_bin, strictstepout=False)                
            CurrJunc.gen_pointsanglecount()
            CurrJunc.analyze_pointsanglecount()#DownStreamSearchPath)
        internalwalknetworkmainpathjunctionset = {}
        for path in CurrJunc.downstreampointandpaths.values():
            if InternalRunTime == 0:
                if not self.is_in_donepathsdict(path, DonePaths):
                    self.add_to_donepathsdict(path, DonePaths, Layer)
                    path.set_streamdirection_by_point(CurrJunc.junctionpoint, True)
                    if path.junctions[1]:
                        internalwalknetworkmainpathjunctionset[path] = path.junctions[1]
        for path in CurrJunc.upstreampointandpaths.values():
            if InternalRunTime == 0:
                if not self.is_in_donepathsdict(path, DonePaths):
                    self.add_to_donepathsdict(path, DonePaths, Layer)
                    path.set_streamdirection_by_point(CurrJunc.junctionpoint, False)
                    if path.junctions[1]:
                        internalwalknetworkmainpathjunctionset[path] = path.junctions[1]
        if InternalRunTime == 0:
            return internalwalknetworkmainpathjunctionset
        elif InternalRunTime == 1:
            return

    def get_longest_path_chain(self, DistanceMode : Literal["AbsStartToAbsEndpoint", "RelStartToRelEndpoint", "TravelDistance"] = "AbsStartToAbsEndpoint"):
        #We loop through endpoints and try to walk the network without turning more that 120°
        for endpoint in self.endpoints:
            match DistanceMode:            
                case "AbsStartToAbsEndpoint":
                    pass
                case "RelStartToRelEndpoint":
                    pass
                case "TravelDistance":
                    pass
    
    def get_furthest_endpoints(self):
        maxdist = 0
        outps = None
        for startp in self.endpoints:
            for endp in self.endpoints:
                if not startp == endp:
                    curdist = point_to_point_dist(startp, endp)
                    if curdist > maxdist:
                        maxdist = curdist
                        outps = [startp, endp]
        return outps
    
    def flip_all(self):
        for junc in self.network:
            junc.flip()

    def save_network(self, savepath = str):
        if os.path.isdir(os.path.basename(savepath)):
            if not savepath.endswith(".pkl"):
                savepath += ".pkl"
        if os.path.exists(savepath):
            os.remove(savepath)
        with open(savepath, 'wb') as handle:
            pickle.dump(self, handle)#, protocol=pickle.HIGHEST_PROTOCOL)

    def collect_all(self, out):
        if not self in out["network"]: out["network"].append(self)
        # dict[junction, None]
        for j in self.network.keys():
            j.collect_all(out)
        for sp in self.singlepaths:  # list[pointpath]
            sp.collect_all(out)
        return out
    
class cluster(BaseObj):
    def __init__(self, clustermask : Optional[np.ndarray], networks : Optional[List[network]]  | None, clusterid : int, name = "cluster", savepath : str = ""):
        super().__init__(name, savepath)
        self.clustermask = clustermask
        self.clusterid = clusterid
        if networks:
            self.networks : Optional[List[network]] = networks
            self.set_networks_parents()
        else:
            self.networks : Optional[List[network]] = []
        
    def set_networks_parents(self):
        for y in self.networks:
            y.parent = self

    def generate_network(self):
        self.networks = [network(self.clustermask, "weird_splitted_skeleton", "TotalClusterNetwork")]

    def split_network(self, network_name, splitallwithname = False):
        x = [y for y in self.networks if (network_name in y.name and splitallwithname) or (y.name == network_name and not splitallwithname)]
        for y in x[::-1]:
            newnets = y.connect_split_process_network_to_networks()
            
            ind = self.networks.index(y)
            self.networks.remove(y)
            
            for i, newnet in enumerate(newnets):
                newnet.name = y.name + "_" + str(i + 1)
                newnet.parent = self
                self.networks.insert(ind, newnet)    

    def collect_all(self, out):
        if not self in out["cluster"]: out["cluster"].append(self)
        for net in self.networks:  # list[network]
            net.collect_all(out)
        return out
    
class clusters(BaseObj):
    def __init__(self, clusterlist : List[cluster] | None, name = "clusters", savepath : str = ""):
        super().__init__(name, savepath)
        if cluster:
            self.cluster : List[cluster] = clusterlist
        else:
            self.cluster : List[cluster] = []
    
    def add_cluster(self, clustermask : np.ndarray, networks : List[network], clusterid : int, name = "Cluster"):
        if name == "Cluster":
            name += + "_Cluster_" + str(len(self.clusters))
            
        self.remove_cluster(name)
        newcluster = cluster(clustermask, networks, clusterid, name)
        self.cluster.append(newcluster)
        return newcluster
        
    def remove_cluster(self,name = "Cluster"):
        a = [x for x in self.cluster if name == x.name]
        if len(a) > 0:
            self.cluster.remove(a[0])
            
    def oldcollect_all(self, out=None):
        if out is None: out = {"clusters": [], "cluster": [], "network": [], "junction": [], "pointpath": []}
        if not self in out["clusters"]:  out["clusters"].append(self)
        for cl in self.cluster:  # list[cluster]
            cl.collect_all(out)
        return out

    def collect_all(self):
        out = {"clusters": [], "cluster": [], "network": [], "junction": [], "pointpath": []}
        if not self in out["clusters"]: out["clusters"].append(self)
        for i, cluster in enumerate(self.cluster):
            if not cluster in out["cluster"]: out["cluster"].append(cluster)
            h, w = cluster.clustermask.shape[0], cluster.clustermask.shape[1]        
            for network in cluster.networks:
                if not network in out["network"]: out["network"].append(network)
                for junction in network.network.keys(): #paths are arranged so that we always go forward!
                    if not self in out["junction"]: out["junction"].append(junction)
                    for ppath in junction.get_all_paths():
                        if not ppath in out["pointpath"]: out["pointpath"].append(ppath)                
                if not len(network.singlepaths) == 0:
                    for ppath in network.singlepaths:
                        if not ppath in out["pointpath"]: out["pointpath"].append(ppath)
        return out
    
    def flowmaps_generate(self, Clusters : Union[Optional[clusters], Tuple[FlowMapGenFiles, clusters, Optional[Queue], Optional[Queue]]]):
        if not clusters:
            Clusters = self
        return flowmaps_generate(Clusters)

    def rename_clusters_members_numeration(self, Clusters : clusters = None):
        #->implementation for working inside of clusters. we want to build names based on network relations for ID
        if Clusters is None:
            Clusters = self
        if not isinstance(Clusters, clusters):
            raise Exception(f"Error in 'rename_network_members_numeration': no valid networks type ({type(Clusters)})")
        
        self.name = f"Clusters"
        self.layername = f"Clusters"
        for h, cluster in enumerate(Clusters.cluster):
            cluster.layername = f"Cluster_{h}"
            cluster.name = f"Cluster_{h}"            
            for i, net in enumerate(cluster.networks):
                pointpathcount = 1
                net.layername = "_".join([cluster.layername, f"Network_{i}"])
                net.name = f"Network_{i}"
                for j, junc in enumerate(net.network.keys()):
                    junc.layername = "_".join([net.layername, f"Junction_{j}"])
                    junc.name = f"Junction_{j}"
                    for k, ppath in enumerate((junc.upstreampointandpaths | junc.downstreampointandpaths).values()):
                        ppath.layername = "_".join([junc.layername, f"Junctionpath_{pointpathcount}"])
                        ppath.name = f"Junctionpath_{pointpathcount}"
                        pointpathcount += 1
                for k, ppath in enumerate(net.singlepaths):
                    ppath.layername = "_".join([net.layername, f"Singlepath_{pointpathcount}"])
                    ppath.name = f"Singlepath_{pointpathcount}"
                    pointpathcount += 1

class flowmapobj(BaseObj):
    def __init__(self, flowmap_image : np.ndarray, panner_image : np.ndarray = None, name = "flowmap", savepath = "", panningUV : Tuple[int,int] = (1.0,1.0), FlowmapGenFile = None):
        super().__init__(name, savepath)
        self.flowmap_image = flowmap_image
        self.panner_image = panner_image
        self.panning_u, self.panning_v = panningUV
        self.FlowmapGenFile = FlowmapGenFile
        #-> we take overall panner_image for every item, saves memory
        #self.panner_image = panner_image
        
def load_obj(path : Optional[str]):
    if path.endswith(".pkl") and os.path.exists(path) and os.path.isfile(path):
        with open(path, 'rb') as handle:
            obj = pickle.load(handle)
        # Links auflösen
        #reg = resolve_reg(reg)
        return obj
    return None

def flowmaps_generate(Clusters : Union[Optional[clusters], Tuple[FlowMapGenFiles, clusters, Optional[Queue], Optional[Queue]]]):
    wastuple = False
    if type(Clusters) == tuple:
        wastuple = True
        CurFlowMapGenFile, Clusters, UpdateQueue, TerminateQueue = Clusters
        Fmgfname = CurFlowMapGenFile.FlowMapFile.name
        Fmgfformat = CurFlowMapGenFile.FlowMapFile.format
        Fmgfroot = CurFlowMapGenFile.FlowMapFile.root
        if type(Clusters) == list:  #means it really is an list of networks            
            cclusters = cwrap()
            cclusters.cluster = [cwrap()]
            cclusters.cluster[0].networks = Clusters
            names = Clusters[0].layername.split("_")
            cclusters.name = "_".join([names[0], names[1]])
            cclusters.layername = "_".join([names[0], names[1]])
            cclusters.cluster[0].name = "_".join([names[2], names[3]])
            cclusters.cluster[0].layername = "_".join([cclusters.layername, cclusters.cluster[0].name])
            Clusters = cclusters
            
            def collect_all(self : clusters):
                js : List[junction] = [j for j in self.cluster[0].networks[0].network.keys()]
                ppt = []
                for j in js:
                    for pp in j.get_all_paths():
                        if not pp in ppt:
                            ppt.append(pp)
                for pp in self.cluster[0].networks[0].singlepaths:
                    if not pp in ppt:
                        ppt.append(pp)
                out = {"clusters": [self], "cluster": [self.cluster[0]], "network": [self.cluster[0].networks], "junction": js, "pointpath": ppt}
                return out
            
            Clusters.collect_all = MethodType(collect_all, Clusters)
    else:
        Clusters = Clusters
    
    nclusters = Clusters.collect_all()
    progressval = 0.0
    donepaths = set()
    if not wastuple:
        outfmo = []
    for cluster in Clusters.cluster:
        #print(f"cluster:({cluster.name})")      
        for network in cluster.networks:
            #print(f"network:({network.name})")
            ppaths = []
            for junction in network.network.keys(): #paths are arranged so that we always go forward!
                #print(f"junction:({junction.name}), junctionpoint:({junction.junctionpoint})")
                ppaths += [ppath for ppath in junction.get_all_paths() if not ppath in ppaths and not ppath in donepaths]
            donepaths.update(ppaths)
            ppathlist = [p.points for p in ppaths]
            if not len(ppathlist) == 0:
                pathflowmap = flowmap_from_skeletons(polylines=ppathlist,
                                                        clustermask=network.clustermask,
                                                        rdp_eps=4.5,
                                                        samples_per_seg=32,
                                                        max_ctrl_pts=16,
                                                        skeleton_stamp_radius=1,
                                                        skeleton_mask=network.skeleton,
                                                        DebugOutputImages=False)
                                                            #, ShowPaths=False)
                fmoname = "_".join([cluster.name, network.name, "junctions"])
                fmo = flowmapobj(pathflowmap, None, fmoname, Fmgfroot + Fmgfname.replace(Fmgfformat, "_" + fmoname + Fmgfformat), FlowmapGenFile=CurFlowMapGenFile)
                if wastuple:
                    UpdateQueue.put(fmo)                
                    progressval = add_progress(UpdateQueue, progressval, len(ppathlist) / len(nclusters["pointpath"]) * 100)
                else:
                    outfmo.append(fmo)
                
            #progressval = add_progress(UpdateQueue, progressval, 1 / len(nclusters["junction"])/10)
            if not len(network.singlepaths) == 0:
                ppaths = [x for x in network.singlepaths if not x in donepaths]                
                donepaths.update(ppaths)
                ppathlist = [p.points for p in ppaths]
                pathflowmap = flowmap_from_skeletons(polylines=ppathlist,
                                                        clustermask=network.clustermask,
                                                        rdp_eps=4.5,
                                                        samples_per_seg=32,
                                                        max_ctrl_pts=16,
                                                        skeleton_stamp_radius=1,
                                                        skeleton_mask=network.skeleton,
                                                        DebugOutputImages=False)
                                                        #, ShowPaths=False)
                fmoname = "_".join([cluster.name, network.name, "singlepaths"])
                if wastuple: 
                    UpdateQueue.put(flowmapobj(pathflowmap, None, fmoname, Fmgfroot + Fmgfname[:-len(Fmgfformat)] + "_" + fmoname + Fmgfformat, FlowmapGenFile=CurFlowMapGenFile))
                    progressval = add_progress(UpdateQueue, progressval, len(ppathlist) / len(nclusters["pointpath"]) * 100)
                else:
                    outfmo.append(fmo)
            #progressval = add_progress(UpdateQueue, progressval, 1 / len(nclusters["network"]) * 10)
        #progressval = add_progress(UpdateQueue, 1 / len(nclusters["cluster"]) * 100, 0)
    if wastuple:
        UpdateQueue.put(100)
        UpdateQueue.put("End")
    else:
        return outfmo

networkingtypes = Literal["clusters", "cluster", "network", "junction", "junctionpath", "singlepath"]
#def resolve_reg(reg):
#    for o in reg.values():
#        if isinstance(o, Serializable):
#            o.resolve_links(reg)
#    # Top-Level-Objekt zurückgeben (idempotent)
#    return next(iter(reg.values()))

def save_obj(obj, path : Optional[str]):
    if path:
        save_pickle(obj, path)

def save_pickle(obj, path):
    if os.path.isdir(os.path.dirname(path)):
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
        #reg = collect(obj)
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle)

# ----- Hilfsfunktion: Relationen nach dem Laden rekonstruieren -----
#def rewire(objs):
#    lookup = {o.id: o for o in objs}
#    for o in objs:
#        for k, v in list(o.__dict__.items()):
#            if k.endswith("_id") and v in lookup:
#                setattr(o, k[:-3], lookup[v])
#                delattr(o, k)
#            elif k.endswith("_ids"):
#                setattr(o, k[:-4], [lookup[i] for i in v if i in lookup])
#                delattr(o, k)
#    return objs

def merge_small_clusters(labels, min_size=200):
    """
    labels: 2D numpy array (cluster labels)
    min_size: minimale Größe zusammenhängender Pixelkomponenten
    Rückgabe: bereinigte labels
    """
    h, w = labels.shape
    result = labels.copy()
    unique_labels = np.unique(labels)

    # für jedes Cluster separat durchgehen
    for lab in unique_labels:
        if lab == -1:
            continue

        # Maske für aktuelle Cluster-Label
        mask = (result == lab).astype(np.uint8)

        # zusammenhängende Komponenten finden
        comp, ncomp = label(mask)

        for ci in range(1, ncomp + 1):
            comp_mask = (comp == ci)
            comp_size = comp_mask.sum()

            if comp_size < min_size:
                # Nachbarschaft bestimmen
                dilated = np.pad(comp_mask, 1, mode="constant")
                neighbors = set()
                donelbls = []
                # Pixel der Komponente durchgehen
                yy, xx = np.where(comp_mask)
                for y, x in zip(yy, xx):
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neigh_label = result[ny, nx]
                            if neigh_label != lab and not neigh_label in donelbls:
                                neighbors.add(neigh_label)
                                donelbls.append(neigh_label)
                # Entferne -1 aus Nachbarn
                neighbors.discard(-1)

                if len(neighbors) == 0:
                    # kein Nachbar → setze auf -1
                    result[comp_mask] = -1
                elif len(neighbors) == 1:
                    # genau ein Nachbar → merge
                    result[comp_mask] = -1 #neighbors.pop()
                else:
                    # mehrere Nachbarn → merge in größten
                    neigh_counts = Counter(result[result != lab])
                    target = max(neighbors, key=lambda n: neigh_counts[n])
                    result[comp_mask] = target

    return result


def cluster_points(junctions, otherskelbin):
    global currh, currw
    """
    Gruppiert benachbarte Punkte in Clustern und ersetzt sie durch ihren Schwerpunkt.
    Überprüft ob Schwerpunkt auf linie sitzt
    """
    mode = "set"
    if isinstance(junctions, network):
        mode = "network"
        junctions.gen_junctionpointlist()
        points = set(junctions.junctionpointlist)
    else:        
        points = junctions
    mask = np.zeros((currh, currw), np.uint8)
    nodict = True
    if type(points) == dict:
        mode = "dict"
        pointskeys = [p for p in points.keys()]
        nodict = False
    else:
        pointskeys = points
    for (y, x) in pointskeys:
        mask[y, x] = 1
    labeled, n = label(mask)
    centers = center_of_mass(mask, labeled, range(1, n+1))
    if mode == "set": return {(int(round(c[0])), int(round(c[1]))) for c in centers}
    centerpoints = [(int(round(c[0])), int(round(c[1]))) for c in centers]
    
    #Getting better centerpoints while also remembering the 3+ points connecting to the junction
    betterpoints = []
    for (cy, cx) in centerpoints:
        gotmatch = False
        if (cy, cx) in pointskeys:
            betterpoints.append((cy, cx))
            pointskeys.remove((cy, cx))
            gotmatch = True
        else:
            nbs = neighbors(cy, cx, otherskelbin)
            for (py, px) in pointskeys:
                if (py, px) in nbs:
                    betterpoints.append((py, px))
                    pointskeys.remove((py, px))
                    gotmatch = True
                    break
        if not gotmatch:
            raise Exception(f"Bisch dir sicher? cluster points centerpoint hat kein partner!")
    remjunc = []
    #clean up of our network
    for i, junction in enumerate(junctions.network):
        if not junction.junctionpoint in betterpoints:
            remjunc.append(i)
    junctions.poplist(remjunc)
    return junctions


def get_best_nextpoint_from_dir(Dir, Point1, Pointlist, maxAngle = 360):
    BestAngle = 100000
    BestP = None
    for p in Pointlist:
        pdir = get_dir_of_points(Point1, p)
        angle = get_angle_of_dirs(Dir, pdir)
        if angle < BestAngle and angle <= maxAngle:
            BestP = p
            BestAngle = angle
            if angle == 0.0: break
    return BestP

def load_image_seperate_alpha(imgpath : str):

    if not os.path.exists(imgpath):
        better_print(f"{imgpath}: Image not found after: ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfile")
        return None, None
    
    img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)        
    
    #Scaling factor for image size
    CurrImageBiggestSide = np.maximum(img.shape[0], img.shape[1])
    RefImagePropertySizeFactor = CurrImageBiggestSide / RefImageBiggestSide
    
    #Alpha mask + new img without alpha
    alpha_mask = None
    if img.shape[2] >= 4:
        alpha_mask = img[:,:,3].astype(bool)
        img = img[:,:,:3]

    return img, alpha_mask


def cluster_image(img, method : clustering_methods = "meanshift"):
    alpha = None
    if isinstance(img, str):
        img, alpha = load_image_seperate_alpha(img)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if method == "kmeans":
        flat = img.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=clustermethodparameters[method]["Target Clustercount"], n_init=clustermethodparameters[method]["n init"]).fit(flat)
        labels = kmeans.labels_.reshape(h, w)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        clustered = centers[labels]
        return labels, clustered, alpha
    elif method == "meanshift":
        flat = img.reshape(-1, 3).astype(np.float32)
        xbandwidth = estimate_bandwidth(flat, quantile=clustermethodparameters[method]["Quantile"], n_samples=clustermethodparameters[method]["Samples"])
        x = 0
        i = 0
        while x < 3:
            x += 1            
            while xbandwidth == 0.0 and i < 15 * x:
                i += 1
                xbandwidth = estimate_bandwidth(flat, quantile=clustermethodparameters[method]["Quantile"]*1.2*i, n_samples=int(clustermethodparameters[method]["Samples"]*1.1*i))
            if xbandwidth == 0.0:
                return None, None, alpha
            ms = MeanShift(bandwidth=xbandwidth, bin_seeding=True).fit(flat)
            labels = ms.labels_.reshape(h, w)
            if labels.max() != 0:
                break
            
        centers = ms.cluster_centers_.astype(np.uint8)
        clustered = centers[labels]
        return labels, clustered, alpha
    elif method == "slic":
        labels = segmentation.slic(img, n_segments=clustermethodparameters[method]["Target Clustercount"], compactness=clustermethodparameters[method]["Compactness"], start_label=0)
        clustered = color.label2rgb(labels, img, kind='avg').astype(np.uint8)
        return labels, clustered, alpha
    else:
        raise ValueError("Unsupported clustering method")

def visualize_clusters(labels, img_shape, draw_boundaries = False, backgroundcolor = (0,0,0,0)):
    h, w = img_shape[:2]
    vis = np.full((h, w, 4), backgroundcolor, dtype=np.uint8)
    num_clusters = labels.max() + 1
    colors = np.random.randint(15, 255, size=(num_clusters, 3), dtype=np.uint8)
    colors = np.hstack([colors, np.full((num_clusters,1), 255, dtype=np.uint8)])
    for i in range(num_clusters):
        vis[labels == i] = colors[i]
    if draw_boundaries:
        boundaries = segmentation.find_boundaries(labels, mode='thin').astype(np.uint8) * 255
        vis[boundaries == 255] = [0, 0, 0, 255]
    return vis
       
def set_Accessmanager_all():
    AccessMngr.set_AccessGroup({"mainstages" : ["processfiles", "networkclass"]})
    AccessMngr.set_AccessGroup({"substeps" : ["file", "cluster", "networks", "flowmapgen"]})
    AccessMngr.set_AccessGroup({"maptype" : ["skeleton", "clustermask", "depthmap", "flowmap", "junctions", "endpoints", "arrowmap"]})

def set_Accessmanager(Set = True):
    if Set:
        AccessMngr.set_AccessGroup({"mainstages" : ["processfiles", "networkclass"]})
        AccessMngr.set_AccessGroup({"substeps" : ["file", "cluster", "networks", "flowmapgen"]})
        AccessMngr.set_AccessGroup({"maptype" : ["flowmap"]})
    else:
        AccessMngr.set_AccessGroup({"mainstages" : []})
        AccessMngr.set_AccessGroup({"substeps" : []})
        AccessMngr.set_AccessGroup({"maptype" : []})

def merge_flowmap_on_flowmap(LowerLayer : np.ndarray, UpperLayer : np.ndarray):
    ActualLowerLayer = LowerLayer.copy()
    mask = np.all(UpperLayer[..., :3] != 0, axis=-1) | UpperLayer[..., 3] != 0
    ActualLowerLayer[mask] = UpperLayer[mask]
    return ActualLowerLayer

def process_files(files, resetFolder = True, minclustersize = 500):
    
    global outputfolder, file
    
    flowmaparr = [] 
    Init_TimeGroup("totaltime")   
    for file in files:
        outputfolder = baseoutputfolder + "/" + file.name.replace(file.format,"")         
        better_print(f"{file.name}: Starting" , ShowTimeDiff=False, ResetTime=True, TimeGroup="processfile")
        #outputfilebase = file.name[:-len(file.format)]
        
        img = cv2.imread(file.fullname, cv2.IMREAD_UNCHANGED)
        if img is None:
            better_print(f"{file.name}: Image not found after: ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfile")
            continue
        better_print(f"{file.name}: Image loaded after: ", ", starting generation of clusters...", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfile")
        
        #Scaling factor for image size
        CurrImageBiggestSide = np.maximum(img.shape[0], img.shape[1])
        RefImagePropertySizeFactor = CurrImageBiggestSide / RefImageBiggestSide
        
        #Alpha mask + new img without alpha
        alpha_mask = None
        if img.shape[2] >= 4:
            alpha_mask = img[:,:,3].astype(bool)
            img = img[:,:,:3]

        #Clustering
        ClusterImgPath = outputfolder + "/" + file.name.replace(file.format, "_" + output_clusteradder  + "_snap_" + file.format)
        if not LoadLastClusterimg or not (os.path.exists(ClusterImgPath)):
            labels, cluster_img, alpha = cluster_image(img, method=clustering_method) #new_cluster_image(img)            
            del(alpha)
            if AccessMngr.can_access(["processfiles", "cluster"]): SaveFlowmap(outputfolder, file.name, file.format, visualize_clusters(labels, img.shape), output_clusteradder, "process_file")
            SaveFlowmap(outputfolder, file.name, file.format, labels, output_clusteradder + "_snap_")
            better_print(f"{file.name}: process_files 1: Clustered after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfile")
        else:
            labels = cv2.imread(ClusterImgPath, cv2.IMREAD_UNCHANGED).astype(np.int64)
            better_print(f"{file.name}: process_files 1: Clusters loaded from {ClusterImgPath} after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfile")
            if len(labels.shape) > 2:
                if labels.shape[2] > 0:
                    labels = labels[:,:,0]
        
        #removing alpha mask from labels to avoid clustering of those
        if alpha_mask is not None:
            labels[~alpha_mask] = -1  #

        #should be done to fill holes and merge small fragments
        #labels = merge_small_clusters(labels, 200)
        if resetFolder: del_files_of_folder(outputfolder, sourcefileincludefilters, ["_snap_", "_clusters.png"], allowedimagetypes)

        global currh, currw
        currh, currw = img.shape[:2]        
        # pro Cluster arbeiten
        jk = 0
        uniquelabels = [unlb for unlb in np.unique(labels) if not unlb == -1]
        better_print(f"{file.name}: process_files 2: working on {str(len(uniquelabels))} clusters after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfile")
        Init_TimeGroup("process_clusters")
        
        dt, bitmax = get_datatype(16)
        fileflowmap = np.zeros((currh, currw, 4), dt) * bitmax
        for i, cid in enumerate(uniquelabels):
            if cid == -1: continue  #alpha channel
            Init_TimeGroup("process_cluster")
            better_print(f"{file.name}: Cluster ({str(i+1)}/{str(len(uniquelabels))}) Start:", " networking", ShowTimeDiff=False, ResetTime=True, TimeGroup="process_cluster")            
            
            jk += 1
            mask = (labels == cid)
            if AccessMngr.can_access(["processfiles", "cluster", "clustermask"]): FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, mask, output_clusteradder + f"_cluster_{str(jk)}_", "cluster_mask")
            if 1 == 0:
                skel = weird_splitted_skeleton(mask, True, masksmoothmode=["Gaussian"])
                print("AHAWH")
            #"Too small Clusters dont matter"
            
            if not np.count_nonzero(mask) >= minclustersize:
                better_print(f"{file.name}: Cluster ({str(i+1)}/{str(len(uniquelabels))}) , process_files 2.2: {str(np.count_nonzero(mask))} points in cluster -> skipping... , border is {str(200 * RefImagePropertySizeFactor)}", "", False, True, "process_cluster")
                continue
            else:
                better_print(f"{file.name}: Cluster ({str(i+1)}/{str(len(uniquelabels))}) , process_files 2.1: {str(np.count_nonzero(mask))} points in cluster to work on... , border is {str(200 * RefImagePropertySizeFactor)}", "", False, True, "process_cluster")
            
            #netzwerke generieren
            
            networks = network(mask, SkeletonType)
            if AccessMngr.can_access(["processfiles", "networks", "junctions"]): SaveFlowmap(outputfolder, file.name, file.format, PointsToFlowmap(networks.junctionpointlist, currh, currw, 1, radius = 0), f"_label({str(jk)})_junctionpointlist_b4_sortnet", "process_juncs")
            networks = networks.connect_split_process_network_to_networks()
            
            better_print(f"{file.name}: Cluster ({str(i+1)}/{str(len(uniquelabels))}) , Networked using skeletize-method ({SkeletonType}) for ", f", got {len(networks)} seperate networks out of cluster. Starting generation of Flowmap...", ShowTimeDiff=True, ResetTime=True, TimeGroup="process_clusters")
            clusterflowmap = np.zeros((currh, currw, 4), dt) * bitmax
            Init_TimeGroup("networksgenerated")
            for af, networ in enumerate(networks):
                networkflowmap = np.zeros((currh, currw, 4), dt) * bitmax  # 127 = kein Flow
                pointpaths = []
                Init_TimeGroup("networkgenerated")
                if AccessMngr.can_access(["processfiles", "networks", "clustermask"]): SaveFlowmap(outputfolder, file.name, file.format, networ.clustermask.astype(np.uint8) * 255, output_flowmapadder + f"_cluster_{str(jk)}_network_{str(af)}_clustermask", "process_network_cluster")
                pointpaths = networ.get_network_pointpaths()
                ppathlist = [ppath.points for ppath in pointpaths]
                
                networkflowmap = flowmap_from_skeletons(polylines=ppathlist,
                                                        clustermask=networ.clustermask,
                                                        rdp_eps=4.5,
                                                        samples_per_seg=32,
                                                        max_ctrl_pts=16,
                                                        skeleton_stamp_radius=1,
                                                        skeleton_mask=networ.skeleton,
                                                        DebugOutputImages=False)
                                                        #, ShowPaths=False) 
                
                clusterflowmap = merge_flowmap_on_flowmap(clusterflowmap, networkflowmap)                
                better_print(f"{file.name}: Cluster ({str(i+1)}/{str(len(uniquelabels))}) network {str(af+1)}/{str(len(networks))} , generated flowmap for ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="networkgenerated")
                if AccessMngr.can_access(["processfiles", "networks", "flowmap"]): FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, networkflowmap, output_flowmapadder + f"_cluster_{str(jk)}_network_{af}_", "process_network_flowmap")
            
            fileflowmap = merge_flowmap_on_flowmap(fileflowmap, clusterflowmap)            
            if FlowmapFilePath: flowmaparr.append(xfile(FlowmapFilePath))
            if AccessMngr.can_access(["processfiles", "cluster", "flowmap"]): FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, clusterflowmap, output_flowmapadder + f"_cluster_{str(jk)}_allnetworks_", "process_cluster_flowmap")
            better_print(f"{file.name}: Cluster ({str(i+1)}/{str(len(uniquelabels))}) - all networks generated, took ", "", True, True, "process_clusters")
            
        if AccessMngr.can_access(["processfiles", "file", "flowmap"]):FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, fileflowmap, output_flowmapadder + f"_cluster_{str(jk)}_networks_file_total_", "process_file_flowmap")
        
        better_print(f"##FILE ({file.name}) FINISH## after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file")
    better_print(f"##TOTAL FINISH## after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="totaltime")            
            
    #import flowmap_vektor_length_unionizer as fvlu
    #flowmaps = fvlu.process_files(flowmaparr)
    #import os
    #os.system('shutdown /p /f') 



def set_file_and_currhw(ogfile : xfile, path : str):
    global currh, currw, file
    file = ogfile
    img, alphamask = load_image_seperate_alpha(path)
    currh, currw = img.shape[0], img.shape[1]
    return img

def cluster_method_reload_run(UpdateQueue : Queue, TerminateQueue : Queue, path : str, ogfile : xfile, method : str, clustermethparamaters, xterminatequeue : Queue, fntype : str):
    global clustermethodparameters
    clustermethodparameters = clustermethparamaters
    img = set_file_and_currhw(ogfile=ogfile, path=path)
    UpdateQueue.put(randint(1,99))
    labels, clusterimg, alpha = cluster_image(img, method)
    UpdateQueue.put((path, labels, clusterimg, alpha))
    xterminatequeue.put(fntype)

def network_generate_run(UpdateQueue : Queue, TerminateQueue : Queue, path : str, ogfile : xfile, tblit : clusters, xterminatequeue : Queue, fntype : str):
    #try:
        #reg = collect(tblit)
        #tblit.resolve_links(reg)
    set_file_and_currhw(ogfile=ogfile, path=path)
    for i, cluster in enumerate(tblit.cluster):
        #print(f"network_generate_run: starting cluster {i}")
        newnetwork = network(cluster.clustermask, SkeletonType, "Network_1")
        #newnetwork.append(newnetwork)
        #newnetwork = collect(newnetwork)
        cluster.networks = [newnetwork]
        #UpdateQueue.put((path, newnetwork, i))
        #UpdateQueue.put(min(99, (i+1) / len(tblit.cluster) * 50))
        #print(f"network_generate_run: end cluster {i}")
    network_split_run(UpdateQueue=UpdateQueue, TerminateQueue=TerminateQueue, cclusters=tblit, ogfile = ogfile, path = path, xterminatequeue=xterminatequeue, fntype=fntype)
    UpdateQueue.put("End")
    #xterminatequeue.put(fntype)

def network_split_run(UpdateQueue : Queue, TerminateQueue : Queue, cclusters : clusters, ogfile : xfile, path : str, xterminatequeue : Queue, fntype : str):
    set_file_and_currhw(ogfile=ogfile, path=path)
    #cclusters : clusters = resolve_reg(cclusters)
    for i, clust in enumerate(cclusters.cluster):
        netw = clust.networks[0]
        #print(f"network_split_run: starting network {i}")
        #try:    
        newnetworks = netw.connect_split_process_network_to_networks()
        #except Exception as e:
        #    #print(e)
        #    UpdateQueue.put(Exception(f"network_split_run->connect_split_process_network_to_networks got Exception: ({e})"))
        #    UpdateQueue.put(min(99, i / len(cclusters.cluster) * 50 + 50))
        #    continue
        #newnetworks = collect(newnetworks)
        clust.networks = newnetworks        
        UpdateQueue.put(min(99, i / len(cclusters.cluster) * 50 + 50))
        #print(f"network_split_run: end network {i}")
    UpdateQueue.put(cclusters)
    UpdateQueue.put("End")
    
    xterminatequeue.put(fntype)

def add_progress(UpdateQueue : Queue, progressval, adder) -> int | float:
    progressval += adder
    UpdateQueue.put(progressval)
    return progressval

def flowmap_generate_all_run(UpdateQueue : Queue, TerminateQueue : Queue, Clusters : clusters, ogfile : xfile, path : str, xterminatequeue : Queue, fntype : str):
    set_file_and_currhw(ogfile=ogfile, path=path)
    try:
        flowmaps_generate(Clusters=Clusters + (UpdateQueue, TerminateQueue))
        add_progress(UpdateQueue, 100, 0)
    except Exception as e:
        UpdateQueue.put(Exception(f"Error in flowmap_generate_all_run: ({e})"))
        add_progress(UpdateQueue, 0, 0)
        #print(f"Error while flowmap_generate_all_run: ({e})")
    xterminatequeue.put(fntype)
    
def flowmap_generate_single_run(UpdateQueue : Queue, TerminateQueue : Queue, Clusters : clusters, ogfile : xfile, path : str, xterminatequeue : Queue, fntype : str):
    set_file_and_currhw(ogfile=ogfile, path=path)
    try:
        flowmaps_generate(Clusters=Clusters + (UpdateQueue, TerminateQueue))
        add_progress(UpdateQueue, 100, 0)
    except Exception as e:
        e = Exception(f"Error in flowmap_generate_single_run: ({e})")
        UpdateQueue.put(e)
        add_progress(UpdateQueue, 0, 0)
        #print(f"Error while flowmap_generate_single_run: ({e})")
    xterminatequeue.put(fntype)

if __name__ == "__main__":
    set_Accessmanager()
    process_files(files)

#-> cute parameter mode: Literal["angle", "neighbourcount", "neighbourdir", "neighbourdirold"] = "neighbourcount"