 #-uses:
#-clustering for meanshift
#-per-cluster-skeleton
# ->contours
# ->direction of contours should be similar
# ->contour to spline
# ->spline searches border
# ->fill from last borderpoints to next


from typing import Literal, Tuple, Optional, List, Set
import cv2, random, os

import numpy as np

from skimage import morphology, segmentation, color
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth


from settings import *
from scipy import cluster, interpolate
from scipy.interpolate import splprep, splev
from scipy.ndimage import label, center_of_mass, find_objects
from scipy.spatial import cKDTree
from collections import Counter
import matplotlib.pyplot as plt

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


def _arclength_parameterize(X):
    # arclength-s -> u in [0,1]
    dif = np.diff(X, axis=0)
    seg = np.hypot(dif[:,0], dif[:,1])
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] == 0:
        return np.linspace(0,1,len(X))
    return s/s[-1]

def smooth_path(points, smoothing=SMOOTHING_S, num_samples=SAMPLES_PER_PATH):
    # --- Spline-Fitting (C^2), arclength-uniformes Resampling ---
    pts = np.asarray(points, dtype=float)
    # robust gegen Duplikate:
    _, idx = np.unique(pts, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    if len(pts) < 4:  # weniger als 4 → kubischer Spline nicht möglich
        # lineare Interpolation
        u = np.linspace(0, 1, len(pts))
        unew = np.linspace(0, 1, max(num_samples, len(pts)))
        x = np.interp(unew, u, pts[:,0])
        y = np.interp(unew, u, pts[:,1])
        return np.column_stack([x, y])

    # param. Spline
    tck, u = splprep([pts[:,0], pts[:,1]], s=smoothing)
    # zunächst dicht sampeln, dann auf arclength re-samplen
    uu = np.linspace(0,1, max(num_samples*4, 200))
    x_, y_ = splev(uu, tck)
    XY = np.column_stack([x_, y_])
    su = _arclength_parameterize(XY)
    unew = np.linspace(0,1,num_samples)
    sm = np.column_stack([np.interp(unew, su, XY[:,0]),
                          np.interp(unew, su, XY[:,1])])
    return sm

def tangent_and_normal(path):
    # --- Tangente/Normalen (normiert) ---
    P = np.asarray(path)
    d = np.gradient(P, axis=0)
    L = np.linalg.norm(d, axis=1, keepdims=True)
    L[L==0] = 1.0
    T = d / L
    N = np.column_stack([-T[:,1], T[:,0]])
    return T, N

# -----------------------------------
# Daten aus Junctions einsammeln
# -----------------------------------

def collect_unique_pointpaths(junctions):
    # --- alle PointPaths aus up/down dicts, Duplikate vermeiden ---
    seen = set()
    paths = []

    for j in junctions:
        for dct in (j.upstreampointandpaths, j.downstreampointandpaths):
            for key, pp in dct.items():
                # Schlüssel: id des Objekts, falls verfügbar; sonst Endpunkte
                try:
                    key_id = id(pp)
                except:
                    key_id = (tuple(pp.points[0]), tuple(pp.points[-1]), len(pp.points))
                if key_id in seen:
                    continue
                seen.add(key_id)
                # Sicherheit: Richtung beibehalten (points[0] ist upstream)
                pts = [(float(x), float(y)) for (x,y) in pp.points]
                if len(pts) >= 2:
                    paths.append(pts)
    return paths

# -----------------------------------
# Strömungsgrößen (Breite/Tiefe/Abfluss)
# -----------------------------------

def assign_section_props(num, base_width=BASE_WIDTH, base_depth=BASE_DEPTH, Q=1.0):
    # --- einfache Profile: Breite/Tiefe entlang s, optional Glocke in der Mitte ---
    s = np.linspace(0,1,num)
    width = base_width * (0.85 + 0.3*np.exp(-((s-0.5)/0.35)**2))  # sanfte Aufweitung mittig
    depth = base_depth * (0.9 + 0.2*np.exp(-((s-0.5)/0.4)**2))    # etwas tiefer mittig
    # Kontinuität: u = Q / (b*h); clamp
    vel = Q / (width * depth + 1e-8)
    return width, depth, vel

# -----------------------------------
# Samples aggregieren & KD-Tree
# -----------------------------------

def sample_paths_build_tree(paths, per_path_Q=None):
    # --- alle Pfade glätten + Tangente/Normalen + Breite/Tiefe/Velocity bestimmen ---
    all_pos = []
    all_tan = []
    all_w = []
    all_h = []
    all_u = []
    all_sigma = []

    for idx, pts in enumerate(paths):
        sm = smooth_path(pts)
        T, N = tangent_and_normal(sm)
        # Abschnitts-Eigenschaften
        Qp = 1.0 if per_path_Q is None else per_path_Q.get(idx, 1.0)
        w, h, u = assign_section_props(len(sm), Q=Qp)

        # lokale σ (Querprofil) ~ Breite
        sigma = KERNEL_SIGMA_FACT * w

        all_pos.append(sm)
        all_tan.append(T)
        all_w.append(w)
        all_h.append(h)
        all_u.append(u)
        all_sigma.append(sigma)

    P = np.vstack(all_pos)                # (N,2)
    T = np.vstack(all_tan)                # (N,2)
    W = np.concatenate(all_w)             # (N,)
    H = np.concatenate(all_h)             # (N,)
    U = np.concatenate(all_u)             # (N,)
    SIG = np.concatenate(all_sigma)       # (N,)

    tree = cKDTree(P)
    return P, T, W, H, U, SIG, tree

# -----------------------------------
# Vektorfeld-Rasterisierung (Überlagerung an Kreuzungen)
# -----------------------------------

def vector_field_on_grid(P, T, U, SIG, tree, thisnetwork, bounds=None, grid_res=GRID_RES):
    # --- Grid aufspannen ---
    if bounds is None:
        H, W = thisnetwork.skeleton.shape  # Nutze Skeleton-Größe
        xmin, xmax = 0, W-1
        ymin, ymax = 0, H-1
        pad = max(grid_res*5, 0.05*(max(xmax-xmin, ymax-ymin)))
        xmin -= pad; ymin -= pad; xmax += pad; ymax += pad
    else:
        xmin, xmax, ymin, ymax = bounds

    xs = np.arange(xmin, xmax+grid_res, grid_res)
    ys = np.arange(ymin, ymax+grid_res, grid_res)
    X, Y = np.meshgrid(np.arange(xmin, xmax+1, 1),
                   np.arange(ymin, ymax+1, 1))

    Ugrid = np.zeros_like(X, dtype=float)
    Vgrid = np.zeros_like(Y, dtype=float)

    # --- Kernel-Zusammenführung (anisotrope Gauß-Gewichtung je Sample) ---
    # Für jedes Gitterpixel: Nachbarn im Radius r = NEIGHBOR_RADIUS_FACT * σ_i (per-sample)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    # grober globaler Radius für erste Nachbarsuche (min SIG als untere Schranke vermeiden)
    global_sigma = max(1e-3, np.median(SIG))
    base_r = NEIGHBOR_RADIUS_FACT * global_sigma

    # Batchweise arbeiten (Speicher)
    B = 8192
    for i0 in range(0, len(XY), B):
        sl = slice(i0, min(i0+B, len(XY)))
        pts = XY[sl]
        # Vorselektion mit grobem Radius
        neigh_lists = tree.query_ball_point(pts, r=base_r)
        # Beitrag jedes Nachbarn addieren
        uacc = np.zeros(len(pts))
        vacc = np.zeros(len(pts))
        wsum = np.zeros(len(pts))

        for j, neigh in enumerate(neigh_lists):
            if not neigh:
                continue
            q = pts[j]
            nn = np.asarray(neigh, dtype=int)
            Pn = P[nn]
            Tn = T[nn]
            Un = U[nn]
            sig = SIG[nn]
            # exakter Radius je Sample
            d2 = np.sum((Pn - q)**2, axis=1)
            # Gewichte (anisotrop quer zur Bahn ~ σ): nähern mit isotropem Gauß in 2D
            w = np.exp(-0.5 * d2 / (sig**2 + 1e-12))
            # Richtungsbeitrag = Geschwindigkeit * Tangente * Gewicht
            contrib = (Un * w)[:,None] * Tn
            uacc[j] = contrib[:,0].sum()
            vacc[j] = contrib[:,1].sum()
            wsum[j] = w.sum()

        # Normalisieren optional (stabilere Richtung, konserviert nicht die Stärke strikt)
        nz = wsum > 1e-12
        Ugrid.ravel()[sl][nz] = uacc[nz] / wsum[nz]
        Vgrid.ravel()[sl][nz] = vacc[nz] / wsum[nz]
        # (wo keine Nachbarn → 0)

    return X, Y, Ugrid, Vgrid

# -----------------------------------
# Streamlines & (einfaches) LIC
# -----------------------------------

def plot_streamlines(X, Y, U, V, density=1.4, linewidth=1.0, skeleton = None):
    # --- Streamlines basierend auf dem Vektorfeld ---
    fig, ax = plt.subplots(figsize=(10, 8))
    speed = np.hypot(U, V)
    ax.set_facecolor((0/255, 127/255, 127/255))
    ax.streamplot(X, Y, U, V, density=1.2, color=np.hypot(U,V), linewidth=1.0)
    H, W = skeleton.shape if skeleton is not None else (X.shape[0], X.shape[1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # origin upper → y-Achse invertieren
    ax.set_aspect('equal')
    ax.set_aspect('equal', 'box')
    ax.set_title('Flowmap (Streamlines)')
    return fig, ax

def lic_simple(U, V, length=12, steps=20, noise=None, seed=0):
    # --- sehr einfache LIC-Approx (Cabral & Leedom-ähnlich, Euler-Vorschritt) ---
    # Hinweis: Für Produktion ggf. echte LIC-Implementierung/Lib nutzen.
    rng = np.random.default_rng(seed)
    H, W = U.shape
    if noise is None:
        noise = rng.random((H, W)).astype(np.float32)
    # normierte Richtung
    mag = np.hypot(U, V); mag[mag==0] = 1.0
    Uu, Vv = U/mag, V/mag

    out = np.zeros_like(noise, dtype=np.float32)
    # Trace vorwärts + rückwärts, akkumulieren
    for sign in (-1.0, 1.0):
        pos_y, pos_x = np.mgrid[0:H, 0:W].astype(np.float32)
        acc = np.zeros_like(noise)
        wsum = np.zeros_like(noise)
        for _ in range(steps):
            # Sample Noise an aktueller Position (bilinear)
            x0 = np.clip(pos_x, 0, W-1); y0 = np.clip(pos_y, 0, H-1)
            x1 = np.clip(x0+1, 0, W-1); y1 = np.clip(y0+1, 0, H-1)
            fx = pos_x - np.floor(x0); fy = pos_y - np.floor(y0)
            n00 = noise[y0.astype(int), x0.astype(int)]
            n01 = noise[y1.astype(int), x0.astype(int)]
            n10 = noise[y0.astype(int), x1.astype(int)]
            n11 = noise[y1.astype(int), x1.astype(int)]
            nxy = (n00*(1-fx)*(1-fy) + n10*fx*(1-fy) + n01*(1-fx)*fy + n11*fx*fy)

            acc += nxy
            wsum += 1.0
            # Schritt entlang Feld
            vx = Uu[np.clip(np.round(pos_y).astype(int),0,H-1),
                    np.clip(np.round(pos_x).astype(int),0,W-1)]
            vy = Vv[np.clip(np.round(pos_y).astype(int),0,H-1),
                    np.clip(np.round(pos_x).astype(int),0,W-1)]
            pos_x += sign * vx * (length/steps)
            pos_y += sign * vy * (length/steps)
        out += np.where(wsum>0, acc/wsum, 0.0)
    # Normalisieren
    out -= out.min()
    if out.max() > 0:
        out /= out.max()
    return out

def plot_lic(X, Y, U, V):
    img = lic_simple(U, V, length=14, steps=24)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set_facecolor((0/255, 127/255, 127/255))
    ax.imshow(img, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
    ax.set_aspect('equal', 'box')
    ax.set_title('Flowmap (LIC)')
    return fig, ax

# -----------------------------------
# Hauptfunktion: von Junctions → Flowmap
# -----------------------------------

def flowmap_from_junctions(junctions, thisnetwork, per_path_Q=None,
                           grid_res=GRID_RES,
                           return_all=False,
                           plot='both'):
    """
    junctions: Iterable deiner Junction-Objekte
    per_path_Q: dict{pfad_index:int -> Abfluss Q}, optional
    plot: 'stream', 'lic' oder 'both'
    """
    # 1) Paths einsammeln (inkl. Kreuzungen überlagern wir später im Feld)
    paths = collect_unique_pointpaths(junctions)

    # 2) Samples + Tangente + Breite/Tiefe + Geschwindigkeit + KD-Tree
    P, T, W, H, U, SIG, tree = sample_paths_build_tree(paths, per_path_Q=per_path_Q)

    # 3) Vektorfeld auf Raster (Überlagerung = Summe/gewichtete Mittelung)
    X, Y, Ugrid, Vgrid = vector_field_on_grid(P, T, U, SIG, tree, thisnetwork, grid_res=grid_res)
    
    figs = []
    if plot in ('stream', 'both'):
        figs.append(plot_streamlines(X, Y, Ugrid, Vgrid)[0])
    if plot in ('lic', 'both'):
        figs.append(plot_lic(X, Y, Ugrid, Vgrid)[0])

    if return_all:
        return dict(paths=paths, samples=P, tangents=T, width=W, depth=H,
                    vel=U, sigma=SIG, grid=(X,Y,Ugrid,Vgrid), figs=figs)
    return figs

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

def sort_pointlist_by_angle_to_pt_and_direction(point : Tuple[int,int], basedirection : int, pointlist : List[Tuple[int,int]] = []):
    angles = {}
    for p in pointlist:
        angles[get_angle_of_dirs(get_dir_of_points(point, p), basedirection)] = p
    sortedangles = angles.keys()
    sortedangles.sort()
    outpoints = []
    for angle in sortedangles:
        outpoints.append(angles[angle])
    return outpoints

class pointpath():
    def __init__(self): 
        #upstream, downstream
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
        if self.junctions[0] == None and self.junctions[1] == None:
            return True
        return False
    
    def is_junction_to_endpoint(self):
        if (self.junctions[0] == None) != (self.junctions[1] == None):
            return True
        return False

    def flip(self):
        self.junctions = self.junctions[::-1]
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

    def set_path_from_junction(self, startjunction, pathdirpoint, network, maxangle = 90):
        self.points = []
        self.junctions[0] = startjunction
        hasjunction = startjunction != None
        if hasjunction:
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
        if hasjunction:
            prev, curr = startjunction.junctionpoint, get_best_nextpoint_from_dir(lastdir, startjunction.junctionpoint, nbs, maxangle)
        else:
            prev, curr = pathdirpoint, get_best_nextpoint_from_dir(lastdir,pathdirpoint, nbs, maxangle)
        keepRunning = True
        self.points = [prev, curr]
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
            prev, curr = curr, nbs
            lastdir = currdir
            currdir = get_dir_of_points(prev, curr)
            if get_angle_of_dirs(currdir, lastdir) > maxangle:                
                if not prev in network.endpoints: network.endpoints.append(prev)
                if not prev in network.doneendpoints: network.doneendpoints.append(prev)
                break

            self.points.append(nbs)
            #first we check if there is any junction nearby. if so, we take it as target.
            if hasjunction:
                if not closebyjunction: 
                    closebyjunction, dist = network.is_point_close_to_junction(nbs, 3)
                    if closebyjunction == startjunction: closebyjunction = None
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

class junction():    
    def __init__(self, junctionpoint, networklist = []):
        self.junctionpoint = junctionpoint
        self.upstreampointandpaths = {}
        self.downstreampointandpaths = {}
        self.junctionpointignoredneighbors = []
        if not type(networklist) == list:
            raise Exception(f"No valid networklisttype!! {type(networklist)}")
        for net in networklist:
            if not type(net) == network:
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
            for junc in juncpath.junctions:
                if junc and not junc in connectedlist:
                    junc.get_all_connected_juncs(connectedlist, Level + 1)
        if Level != 0:
            return connectedlist
        return [junc for junc in connectedlist]

    def add_to_network(self, addnetwork):
        if type(addnetwork) == network:
            if addnetwork in self.networks:
                self.networks.append(addnetwork)

    def flip(self):
        BUupstr = self.upstreampointandpaths
        self.upstreampointandpaths = self.downstreampointandpaths
        self.downstreampointandpaths = BUupstr
    
    def gen_pointpaths(self, alternetwork = None):
        worknetwork = self.networks[0]
        if alternetwork and type(alternetwork) == network: worknetwork = alternetwork
        for upstrpt in self.upstreampointandpaths.keys():
            ppath = pointpath()
            ppath.set_path_from_junction(self, upstrpt, worknetwork)

            for k, pjunc in enumerate(ppath.junctions):
                if pjunc and not pjunc == self:
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

        for dwstrpt in self.downstreampointandpaths.keys():
            ppath = pointpath()
            ppath.set_path_from_junction(self, dwstrpt, worknetwork)
            for k, pjunc in enumerate(ppath.junctions):
                if pjunc and not pjunc == self:
                    #if we found another junction while creating the path
                    #-> we register the path in the other junctions dictionary unless
                    #   the other junction already has a path for this, then we take this path as ours
                    #   and link ourself into the paths junctions
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
                if maindownstream == None:  #initializing
                    maindownstream = point
                    continue
                if len(self.pointsanglecount[point]["greatereq90"]) >  len(self.pointsanglecount[maindownstream]["greatereq90"]):
                    maindownstream = point
        if maindownstream == None:
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

class PathLayers():
        def __init__(self):
            class PathLayer():
                def __init__(self, Layer = 0, Paths : Optional[list[pointpath]] = [], LayerComplete = False):
                    self.Layer = Layer
                    self.Paths = Paths
                    self.LayerComplete = LayerComplete
                def add_path_to_layer(self, Path : pointpath):
                    if not Path in self.Paths: 
                        self.Paths.append(Path)
            self.Layers : Optional[set[PathLayer]] = {}
        
        def get_layer(self, Layer = 0):
            for lay in self.Layers:
                if lay.Layer == Layer:
                    return lay
                
        def add_path_to_layer(self, Path : pointpath, Layer = 0):
            for lay in self.Layers:
                if lay.Layer == Layer:
                    lay.add_path_to_layer(Path)
                    return

class network():
    def __init__(self, clustermask = None):
        if isinstance(clustermask, np.ndarray):
            self.clustermask = clustermask
            self.skeleton = morphology.skeletonize(clustermask, method='zhang')
        else:
            self.clustermask = None
            self.skeleton = None

        self.network = {}
        self.junctionlist : Optional[list[(int,int)]] = []  #
        self.singlepaths : Optional[list[pointpath]] = [] #
        self.skel_bin = None
        self.poppinglist = False
        self.doneendpoints = []
        #we only initialize network if we have a skeleton to work on
        #else it will be an empty network to be filled elsewise
        self.endpoints = []
        self.skel_bin = None
        if type(self.skeleton) != type(None):    
            self.gen_endpoints_and_junctions()
            SaveFlowmap(outputfolder, file.name, file.format, PointsToFlowmap(self.junctionlist, currh, currw, 1, radius = 0), f"_{output_mergeresolve}_aftergenendpAjunc_", "network")


    PatLayers = PathLayers()
    
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
    
    def add_points_as_junctions(self, pointlist = None):
        if type(pointlist) == list or type(pointlist) == set:
            added = False
            for p in pointlist:
                if type(p) == list or type(p) == set or type(p) == tuple:
                    if len(p) == 2:
                        junc = junction(p, [self])
                        if junc.init_junction(self):
                            self.append(junc)
                            added = True
        #we should be good because of strikt filtering rules when initializing the junction
        #self = cluster_points(self, self.skel_bin)
        if added:
            self.gen_junctionlist()
            SaveFlowmap(outputfolder, file.name, file.format, PointsToFlowmap(self.junctionlist, currh, currw, 1, radius = 0), f"_{output_mergeresolve}_junctionlist_afteraddptasjunc_")
            

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
            self.gen_junctionlist()

    def get_junction_from_endpoint(self, Point) -> junction | None:
        for junc in self.network:
            endp = junc.get_path_from_point(Point)
            if not endp == None:
                return junc
        return None
    
    def get_path_from_endpoint(self, Point) -> pointpath | None: 
        for junc in self.network:
            endp = junc.get_path_from_point(Point)
            if not endp == None:
                return endp
        for singp in self.singlepaths:
            if Point in singp.points:
                return singp
        return None
    
    def pop(self, i, poplist = False):
        if type(i) == int:
            self.network.pop([k for k in self.network.keys()][i])
        if type(i) == junction:
            self.network.pop(i)
        if not self.poppinglist: self.gen_junctionlist()
    
    def gen_endpoints_and_junctions(self, newskel = None):
        global currh, currw
        if newskel:
            self.skeleton = newskel
        workskel = self.skeleton        
        SavePolys(outputfolder, file.name, file.format, currh, currw, self.skeleton, 1, f"_{output_mergeresolve}_juncspre_")
        # Falls Konturen gegeben → zu binärem Skeleton-Bild machen
        self.skel_bin = conv_path_to_skelbin(workskel, self.skeleton.shape[0], self.skeleton.shape[1])
        #if isinstance(workskel, (list, tuple)):
        #    self.skel_bin = np.zeros((currh, currw), np.uint8)
        #    cv2.drawContours(self.skel_bin, workskel, -1, 1, 1)
        #else:    
        #    self.skel_bin = (workskel > 0).astype(np.uint8)
        
        # Pixel-Koordinaten
        coords = np.column_stack(np.nonzero(self.skel_bin))
        
        SaveFlowmap(outputfolder, file.name, file.format, self.skel_bin * 255, f"_{output_mergeresolve}_skel2seg_1skelbin_")

        # Knotentypen finden
        # degree berechnen
        degree = {(y,x): len(neighbors(y,x, self.skel_bin)) for y,x in coords}
        endpoints = {p for p, d in degree.items() if d == 1}
        junctions = {p for p, d in degree.items() if d > 2}
        self.endpoints = [p for p in cluster_points(endpoints, self.skel_bin)]
        SaveFlowmap(outputfolder, file.name, file.format, PointsToFlowmap(endpoints, currh, currw, 1, None, 0), f"_{output_mergeresolve}_skel2seg_2_endpoints_")
        self.add_points_as_junctions(junctions)

    def gen_junctionlist(self):
        self.junctionlist = [junc.junctionpoint for junc in self.network]

    def connect_split_process_network_to_networks(self):
        #We loop through junctions and generate pointpaths which lead to other junctions or nothing (ends)
        self.doneendpoints = []
        for junc in self.network:
            junc.gen_pointpaths()
            
        #we loop through endpoints to check if all are taken. 
        # if they are not taken, it is because it's an endpoint-to-endpoint-connection
        
        self.singlepaths = []
        while not len(self.doneendpoints) == len(self.endpoints):
            lastendpointcount = len(self.endpoints)
            for endpoint in self.endpoints:
                if not endpoint in self.doneendpoints:
                    self.doneendpoints.append(endpoint)
                    singlepath = pointpath()
                    singlepath.set_path_from_junction(None, endpoint, self, 360)
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
        return newnetworks

    def gen_endpoints_from_junctionpaths(self):
        self.endpoints = []
        for junc in self.network.keys():
            juncpaths = junc.upstreampointandpaths | junc.downstreampointandpaths
            for juncpath in juncpaths.values():
                if juncpath.junctions[0] == None or juncpath.junctions[1] == None:
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
            for j in [1,-1]:
                singpathnet.endpoints.append(self.singlepaths[i].points[j])
            singpathnet.skeleton = singpathnet.skel_bin.astype(bool)
            singpathnet.clustermask = get_component_mask(self.clustermask, singpathnet.singlepaths[0].points[0])
        #now we seperate by junctions that arent connected to each other
        junctionnetworks = []
        searchjuncs = [junc for junc in self.network]
        #while len(searchjuncs) != 0:
        while 0 < len(searchjuncs):
            junc = searchjuncs[0]
            searchjuncs.remove(junc)
            connectedjuncs = junc.get_all_connected_juncs()
            newnet = network()
            newnet.network = {junc : None for junc in connectedjuncs}
            newnet.clustermask = get_component_mask(self.clustermask, [tnet for tnet in newnet.network.keys()][0].junctionpoint)
            newnet.gen_endpoints_from_junctionpaths()
            SaveFlowmap(outputfolder, file.name, file.format, newnet.clustermask, output_clusteradder + "_test")
            newnet.gen_skel_and_skelbin_from_junctionpaths(*self.skeleton.shape)
            newnet.gen_junctionlist()
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
                CurrPathsJuncs = [junc for junc in CurrPath.junctions if not junc == None]
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
            if IsDownstream and (not DownOrUpStreamSearchPath in CurrJunc.downstreampointandpaths.values()):
                CurrJunc.flip()
            elif not IsDownstream and (not DownOrUpStreamSearchPath in CurrJunc.upstreampointandpaths.values()):
                CurrJunc.flip()
        
        for path in CurrJunc.upstreampointandpaths.values():
            if not path in DonePaths:
                DonePaths.append(path)
                path.set_streamdirection_by_point(CurrJunc.junctionpoint, True)
                for pjunc in path.junctions:
                    if pjunc and not pjunc in DoneJuncs:
                        self.walk_network_by_unused_paths_recursively(pjunc, DoneJuncs=DoneJuncs, DonePaths=DonePaths, DownOrUpStreamSearchPath=path, IsDownstream=True)
                        break

        for path in CurrJunc.downstreampointandpaths.values():
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
        a : pointpath = pointpath()
        b : junction = junction((0,0), [])
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

# ----------------------------------------------------------------------
# Flowmap Generator mit Bezier/Spline-Kurven
# ----------------------------------------------------------------------

def generate_flowmap(network, base_speed=10.0, n_samples=200, alpha = True):
    """
    Erzeugt eine Flowmap (RGB, shape = (y, x, 3)) aus einem gegebenen Netzwerk.
    Stillstand ist (0,127,127).

    Args:
        network : Instanz deiner class network
        base_speed : Skalenfaktor für mittlere Geschwindigkeit (default 10)
        n_samples : Anzahl an Sample-Punkten pro PointPath (je mehr desto glatter)

    Returns:
        flowmap : np.ndarray shape=(y, x, 3), dtype=np.uint8
    """

    y, x = network.skeleton.shape

    # Flowmap initialisieren mit Stillstand
    if alpha:
        flowmap = np.zeros((y, x, 4), dtype=np.uint8)
    else:
        flowmap = np.zeros((y, x, 3), dtype=np.uint8)
    flowmap[:, :, 1] = 127
    flowmap[:, :, 2] = 127

    # ---------------------------------------------------------------
    # Hilfsfunktionen
    # ---------------------------------------------------------------

    def direction_vector(p0, p1):
        v = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([0.0, 0.0])

    def perpendicular(v):
        return np.array([-v[1], v[0]])

    def rdp(points, epsilon=2.0):
        """
        Ramer-Douglas-Peucker Algorithmus zum Vereinfachen einer Polylinie.
        points : array shape (N,2)
        epsilon : Toleranz, größer = stärker vereinfacht
        """
        pts = np.array(points, dtype=float)

        def _rdp(pts):
            if len(pts) < 3:
                return pts
            # Linie Start-Ende
            start, end = pts[0], pts[-1]
            line = end - start
            line_norm = np.linalg.norm(line)
            if line_norm == 0:
                return np.array([start, end])
            # Distanz aller Punkte zur Linie
            d = np.abs(np.cross(line, pts[1:-1]-start) / line_norm)
            idx = np.argmax(d)
            if d[idx] > epsilon:
                res1 = _rdp(pts[:idx+2])
                res2 = _rdp(pts[idx+1:])
                return np.vstack((res1[:-1], res2))
            else:
                return np.vstack((start, end))
        return _rdp(pts)


    def smooth_curve(points, n_samples=200, epsilon=2.0):
        """
        Vereinfachung per RDP + Bezier-Sampling.
        """
        # 1) Vereinfachen
        keypoints = rdp(points, epsilon=epsilon)

        # 2) B-Spline Fit (k=3 = kubisch)
        tck, u = interpolate.splprep([keypoints[:,0], keypoints[:,1]], s=0, k=min(3, len(keypoints)-1))
        unew = np.linspace(0, 1, n_samples)
        out = interpolate.splev(unew, tck)
        return out[0], out[1]   # echte Subpixel-Koordinaten!


    def blend_line(flowmap, p0, p1, color, thickness=1):
        mask = np.zeros(flowmap.shape[:2], dtype=np.uint8)
        cv2.line(mask, (p0[0], p0[1]), (p1[0], p1[1]), 255, thickness=thickness)

        ys, xs = np.where(mask > 0)
        color = np.array(color, dtype=np.int32)  # groß genug für Zwischenwerte
        for y, x in zip(ys, xs):
            old = flowmap[y, x].astype(np.int32)
            if (old[1] != 127 or old[2] != 127):  # schon Farbe drin
                new = (old + color) // 2
            else:
                new = color
            flowmap[y, x] = np.clip(new, 0, 255).astype(np.uint8)
    # ---------------------------------------------------------------
    # 1. Alle pointpaths sammeln
    # ---------------------------------------------------------------
    pointpaths = []

    if len(network.network) > 0:
        for j in network.network.keys():
            for pp in list(j.upstreampointandpaths.values()) + list(j.downstreampointandpaths.values()):
                if not pp in pointpaths: pointpaths.append(pp)
    else:
        pointpaths.extend(network.singlepaths)
    
    # ---------------------------------------------------------------
    # 2. Für jeden pointpath Flow entlang Kurve
    # ---------------------------------------------------------------
    better_print(f"{len(pointpaths)} to work on..", "", False, True, "flowmapgen_pointpaths")
    for i, pp in enumerate(pointpaths):
        better_print(f"{len(pp.points)} points of pointpath ({i + 1}/{len(pointpaths)}).. after ", "", True, True, "flowmapgen_pointpaths")
        pts = pp.points

        # -> glatte Kurve erzeugen
        curve_y, curve_x = smooth_curve(pts)
        curve = np.stack([curve_y, curve_x], axis=1)
        better_print(f"{len(curve)} points after smoothing ", ", beginning to draw", False, False, "flowmapgen_pointpaths")
        
        # -> entlang Kurve iterieren
        for i in range(len(curve)-1):
            p0 = curve[i].astype(int)
            p1 = curve[i+1].astype(int)

            if not (0 <= p0[0] < y and 0 <= p0[1] < x and 0 <= p1[0] < y and 0 <= p1[1] < x):
                continue

            # Richtungsvektor & Normalenvektor
            v = direction_vector(curve[i], curve[i+1])
            n = perpendicular(v)

            # Breite bestimmen: Distanz bis Clustermask-Rand orthogonal
            mid = (p0 + p1) // 2
            width = 0
            while True:
                left = (int(mid[0] + n[0]*width), int(mid[1] + n[1]*width))
                right = (int(mid[0] - n[0]*width), int(mid[1] - n[1]*width))
                if not (0 <= left[0] < y and 0 <= left[1] < x and network.clustermask[left] == 1):
                    break
                if not (0 <= right[0] < y and 0 <= right[1] < x and network.clustermask[right] == 1):
                    break
                width += 1

            half_width = max(1, width)

            # Geschwindigkeit ~ 1/Breite
            speed_center = 127#base_speed / half_width

            # Strömungsprofil orthogonal auffächern
            for w in range(-half_width, half_width+1):
                pos0 = (int(p0[0] + n[0]*w), int(p0[1] + n[1]*w))
                pos1 = (int(p1[0] + n[0]*w), int(p1[1] + n[1]*w))

                if not (0 <= pos0[0] < y and 0 <= pos0[1] < x):
                    continue
                if not (0 <= pos1[0] < y and 0 <= pos1[1] < x):
                    continue
                if network.clustermask[pos0] == 0 or network.clustermask[pos1] == 0:
                    continue

                # Geschwindigkeit je nach Abstand zum Zentrum
                relative_speed = speed_center * (1 - abs(w)/(half_width+1))

                # Umwandlung in RGB Flow-Vektor
                dx, dy = v[1], v[0]
                fx = 127 - int(relative_speed * dx)
                fy = 127 + int(relative_speed * dy)

                if alpha:
                    blend_line(flowmap, (pos0[1], pos0[0]), (pos1[1], pos1[0]), (0, fy, fx, 255) , thickness=1)
                    #cv2.line(flowmap, (pos0[1], pos0[0]), (pos1[1], pos1[0]), (0, fy, fx, 255), thickness=1)
                else:
                    #cv2.line(flowmap, (pos0[1], pos0[0]), (pos1[1], pos1[0]), (0, fy, fx), thickness=1)
                    blend_line(flowmap, (pos0[1], pos0[0]), (pos1[1], pos1[0]), (0, fy, fx) , thickness=1)

    return flowmap


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

def filter_contours_by_mask(contours_input, mask):
    """
    Entfernt Punkte aus Konturen, die nicht innerhalb der Maske liegen.

    contours_input : Liste von np.ndarray (oder Tuple von cv2.findContours)
    mask           : 2D-Binärmaske (0 oder 255)
    """
    h, w = mask.shape
    # Falls direkt Tuple von findContours übergeben wurde
    if isinstance(contours_input, tuple):
        contours = contours_input[0]
    else:
        contours = contours_input
    mask_bool = mask > 0

    filtered_contours = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        in_mask = []
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])  # hier casten
            if 0 <= y < h and 0 <= x < w and mask_bool[y, x]:
                in_mask.append((x, y))
        if in_mask:
            filtered_contours.append(np.array(in_mask, dtype=np.int32).reshape(-1, 1, 2))

    return filtered_contours


def cluster_points(junctions, otherskelbin):
    global currh, currw
    """
    Gruppiert benachbarte Punkte in Clustern und ersetzt sie durch ihren Schwerpunkt.
    Überprüft ob Schwerpunkt auf linie sitzt
    """
    mode = "set"
    if type(junctions) == network:
        mode = "network"
        junctions.gen_junctionlist()
        points = set(junctions.junctionlist)
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

# ────────────────────────────────────────────────────────────────────────────────
# 1) Utils: Bogenlängen-Resampling & Glättung (Spline-Fallback integriert)
# ────────────────────────────────────────────────────────────────────────────────

def bezier_sample(ctrl_pts, step=1.0):
    """Kleine kubische Beziers (max 4 Punkte) entlang einer Kurve samplen."""
    ctrl_pts = np.asarray(ctrl_pts, dtype=np.float32)
    if len(ctrl_pts) < 2:
        return ctrl_pts.copy()

    def bezier_cubic(p0, p1, p2, p3, t):
        return ((1-t)**3)*p0 + 3*((1-t)**2)*t*p1 + 3*(1-t)*(t**2)*p2 + (t**3)*p3

    pts_out = []
    for i in range(0, len(ctrl_pts)-1, 3):  # 3er Schritte → 4 Punkte
        seg = ctrl_pts[i:i+4]
        if len(seg) < 4:  # letzter Rest → lineare Interpolation
            seg = np.vstack([seg, np.repeat(seg[-1][None], 4-len(seg), axis=0)])
        # grobe Länge schätzen
        chord = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
        count = max(2, int(np.ceil(chord / step)))
        for t in np.linspace(0, 1, count, endpoint=False):
            pts_out.append(bezier_cubic(*seg, t))
    pts_out.append(ctrl_pts[-1])
    return np.array(pts_out, dtype=np.float32)

def same_point(p1, p2, tol=3):
    return np.linalg.norm(np.array(p1) - np.array(p2)) <= tol

def same_start_x_end(curve1, curve2, tol, xAndOrOr = False):
    ####Rework
    ss = same_point(curve1["start"], curve2["start"], tol=tol)
    ee = same_point(curve1["end"], curve2["end"], tol=tol)
    se = same_point(curve1["start"], curve2["end"], tol=tol)
    es = same_point(curve1["end"], curve2["start"], tol=tol)
    if not xAndOrOr:
        if (ss and ee):
            return "ss", "ee", True
        if (se and es):
            return "se", "es", True
    else:
        if (ss or ee):
            return "ss" if ss else "", "ee" if ee else "", True
        if (se or es):
            return "se" if se else "", "es" if es else "", True
    return "", "", False


def _moving_average(x, k=7):
    if k <= 1: 
        return x
    pad = np.vstack([x[0:1], x, x[-1:]])
    ker = np.ones((k, 1)) / k
    return np.hstack([cv2.filter2D(pad[:,0:1], -1, ker)[1:-1], 
                      cv2.filter2D(pad[:,1:2], -1, ker)[1:-1]])

def _resample_equal_arclen(pts, ds=1.5):
    """Resample open polyline 'pts' (N×2) mit annähernd gleicher Bogenlänge-Abtastung."""
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    if L[-1] < ds:
        return pts.copy()
    new_L = np.arange(0, L[-1], ds)
    # lineare Interpolation auf Stücklistenparametrisierung
    x = np.interp(new_L, L, pts[:,0])
    y = np.interp(new_L, L, pts[:,1])
    return np.column_stack([x, y])

def get_sharp_angles(curve, sharp_angle_deg):
    # 4) Knickpunkte finden (werden nicht entfernt)
    sharp_idx = set()
    for i in range(1, len(curve)-1):
        v1 = curve[i] - curve[i-1]
        v2 = curve[i+1] - curve[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosang))
        if angle_deg > sharp_angle_deg:
            sharp_idx.add(i)
    return sharp_idx

def douglas_peucker_epsilon(curve, target_step, final_target_pts, sharp_angle_deg):
    sharp_idx = get_sharp_angles(curve, sharp_angle_deg)
    print(f"douglas_peucker_epsilon: {str(len(sharp_idx))} Knickpunkte")
    # 5) Douglas–Peucker mit adaptivem Epsilon, bis Zielpunktzahl erreicht
    eps_dp = target_step  # Startwert
    best = curve
    while len(best) > final_target_pts:
        tmp = cv2.approxPolyDP(curve.reshape(-1,1,2), eps_dp, False).reshape(-1,2)
        # Knickpunkte reinmischen
        for idx in sorted(sharp_idx):
            if idx < len(curve):
                pt = curve[idx]
                if not any(np.linalg.norm(pt - p) < 1e-3 for p in tmp):
                    tmp = np.vstack([tmp, pt])
        # Sortieren nach ursprünglicher Reihenfolge
        order = [np.argmin(np.linalg.norm(curve - p, axis=1)) for p in tmp]
        tmp = tmp[np.argsort(order)]
        best = tmp
        eps_dp += 0.5  # Schrittweise aggressiver
        if eps_dp > CurrImageBiggestSide/4:  # Sicherheit
            break
    return best

def simplify_and_smooth(contour, epsilon=1.2, smooth_strength=5, target_step=1.5,
                        final_target_pts=20, sharp_angle_deg=140):
    """Vereinfachung mit Beibehaltung von Knickpunkten."""
    poly = contour.reshape(-1, 2).astype(np.float32)

    # 1) Vorvereinfachung
    approx = cv2.approxPolyDP(poly.reshape(-1,1,2), epsilon, False).reshape(-1,2)
    if len(approx) < 3:
        approx = poly
    print(f"simplyfiyandsmoot 1: {str(len(approx))} approx")
    # 2) Glätten (Spline bevorzugt)
    try:        
        k = 3 if len(approx) > 3 else 1
        tck, u = splprep([approx[:,0], approx[:,1]], s=smooth_strength, k=k)
        u_dense = np.linspace(0, 1, max(50, len(approx)*3))
        x_s, y_s = splev(u_dense, tck)
        smooth = np.column_stack([x_s, y_s]).astype(np.float32)
    except Exception:
        smooth = _moving_average(approx, k=max(3, int(smooth_strength)|1)).astype(np.float32)

    print(f"simplyfiyandsmoot 2: {str(len(smooth))} smooth")
    # 3) Gleichmäßiges Resampling
    curve = _resample_equal_arclen(smooth, ds=target_step).astype(np.float32)

    print(f"simplyfiyandsmoot 3: {str(len(curve))} resample")
    # 4) Knickpunkte finden (werden nicht entfernt)    

    best = douglas_peucker_epsilon(curve, target_step, final_target_pts, sharp_angle_deg)

    print(f"simplyfiyandsmoot 5: {str(len(best))} douglas peucker")
    return best.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────────
# 2) Orthogonales „Wurm“-Füllen pro Segment mit konvexen Quads
# ────────────────────────────────────────────────────────────────────────────────

def _inside(img_shape, p):
    h, w = img_shape[:2]
    return 0 <= p[0] < w and 0 <= p[1] < h

def _march_to_boundary(p_xy, n_xy, labels, start_lab, alpha_mask, max_step=2048):
    global lastboundarymarchdist, initiallastboundarymarchdist, maxboundarymarchdist
    """Ray-March entlang Normalen n_xy (Länge egal, wird normalisiert).
       Gibt Randpunkt innerhalb des Clusters zurück (float)."""
    n = np.array(n_xy, dtype=np.float32)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        return p_xy.copy()
    n /= n_norm

    h, w = labels.shape
    x0, y0 = float(p_xy[0]), float(p_xy[1])
    step = 0
    last_good = np.array([x0, y0], dtype=np.float32)

    while step < max_step:        
        x = x0 + n[0] * step
        y = y0 + n[1] * step
        ix, iy = int(round(x)), int(round(y))
        if 1 == 0:
            #We leave when traveling too far compared to last time
            if not lastboundarymarchdist == initiallastboundarymarchdist and (step > (1 + boundarytolerance) * lastboundarymarchdist * RefImagePropertySizeFactor):
                break
            #We leave when traveling too far to avoid overdrawing
            if step > maxboundarymarchdist * RefImagePropertySizeFactor:
                break
        #we leave when leaving the img
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            break
        #we we leave when leaving the cluster
        if labels[iy, ix] != start_lab:
            break
        #we leave when leaving the cluster
        if alpha_mask is not None and not alpha_mask[iy, ix]:
            break
        last_good[0], last_good[1] = x, y
        step += 1
    lastboundarymarchdist = step
    return last_good

def _segment_ribbon(flowmap, labels, p0, p1, flow_color, alpha_mask):
    """Füllt ein konvexes Quad um das Kurvensegment p0->p1 mit variabler Breite."""
    # Tangente & Normale
    t = p1 - p0
    nrm = np.linalg.norm(t)
    if nrm < 1e-6:
        return
    t /= nrm
    n = np.array([-t[1], t[0]], dtype=np.float32)

    h, w = labels.shape
    # Orientierung der Normalen so wählen, dass +n ins Cluster zeigt
    c0 = (int(round(p0[0])), int(round(p0[1])))
    c1 = (int(round(p1[0])), int(round(p1[1])))
    if not _inside((h, w), c0) or not _inside((h, w), c1):
        return
    lab0 = labels[c0[1], c0[0]]
    test = p0 + n  # 1px Probe
    ti = (int(round(test[0])), int(round(test[1])))
    if not _inside((h, w), ti) or labels[ti[1], ti[0]] != lab0:
        n *= -1  # umdrehen, damit +n in der Fläche bleibt

    # Bis zum Rand marchen (beide Seiten)
    L0 = _march_to_boundary(p0,  n, labels, lab0, alpha_mask)
    R0 = _march_to_boundary(p0, -n, labels, lab0, alpha_mask)
    L1 = _march_to_boundary(p1,  n, labels, lab0, alpha_mask)
    R1 = _march_to_boundary(p1, -n, labels, lab0, alpha_mask)

    # Falls das Segment praktisch keine Breite hat, überspringen
    #if (np.linalg.norm(L0 - R0) < 0.75) and (np.linalg.norm(L1 - R1) < 0.75):
    #    return

    quad = np.array([L0, L1, R1, R0], dtype=np.int32)
    # Maske für das aktuelle Quad erzeugen
    mask = np.zeros(flowmap.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)

    # Pixel innerhalb der Maske
    inside_y, inside_x = np.where(mask == 255)
    for y, x in zip(inside_y, inside_x):
        if labels[y, x] != lab0:
            continue
        old = flowmap[y, x]
        if np.all(old == (0, 127, 127, 255)):  # Hintergrundfarbe
            flowmap[y, x] = flow_color
        else:
            flowmap[y, x] = ((old.astype(np.int32) + flow_color) // 2).astype(np.uint8)


def draw_worm_ribbon(flowmap, labels, curve, alpha_mask, color_from_tangent=True, base_color=(127, 127, 127, 255), clusterindex = 0):
    global lastboundarymarchdist, initiallastboundarymarchdist
    """
    Zeichnet entlang 'curve' (N×2 float) einen geschlossenen 'Wurm' aus kleinen Quads.
    color_from_tangent=True: pro Segment Richtung in Farbe enkodieren (wie bisher).
    """
    #if len(curve) < 2:
    #    return
    H, W = labels.shape
    lastboundarymarchdist = initiallastboundarymarchdist
    print(f"CI[{str(clusterindex)}], draw_worm_ribbon 1: {len(curve)} points in curve to work on...")
    pts = bezier_sample(curve, step=0.25)  # oder Catmull-Rom
    print(f"CI[{str(clusterindex)}], draw_worm_ribbon 2: {len(pts)} points in curve to work on after bezier sample...")
    
    for i in range(len(pts)-1):
        p0 = pts[i]
        p1 = pts[i+1]

        # Segmentfarbe (Richtung auf X/Z, Y in Mittel 127)
        if color_from_tangent:
            t = p1 - p0
            nrm = np.linalg.norm(t)
            if nrm < 1e-6:
                continue
            t /= nrm
            red = int(127 + t[1]*127)
            green = int(127 - t[0]*127)
            flow_color = (0, red , green, 255)
        else:
            flow_color = base_color

        
        _segment_ribbon(flowmap, labels, p0, p1, flow_color, alpha_mask)

def orient_cluster_lines(lines):
    # Graph bauen
    G = {}
    for i, ln in enumerate(lines):
        s = tuple(np.round(ln["start"]).astype(int))
        e = tuple(np.round(ln["end"]).astype(int))
        G.setdefault(s, []).append((e, i))
        G.setdefault(e, []).append((s, i))
    # Hauptader finden (hier: längster Endpunkt-Endpunkt-Pfad)
    endpoints = [p for p in G if len(G[p]) == 1]
    start = endpoints[0]
    end   = max(endpoints, key=lambda p: np.linalg.norm(np.array(p)-np.array(start)))
    # BFS für Orientierung
    visited_edges = set()
    stack = [(start, None)]
    parent_dir = {}
    while stack:
        node, from_idx = stack.pop()
        for neighbor, idx in G[node]:
            if idx in visited_edges: 
                continue
            visited_edges.add(idx)
            # Ausrichtung prüfen
            if tuple(np.round(lines[idx]["start"]).astype(int)) != node:
                lines[idx]["curve"] = lines[idx]["curve"][::-1]
                lines[idx]["start"], lines[idx]["end"] = lines[idx]["end"], lines[idx]["start"]
            stack.append((neighbor, idx))
    return lines
# ────────────────────────────────────────────────────────────────────────────────
# 3) NEUER Block: ersetzt deinen bisherigen 'for poly in contours' Teil
# ────────────────────────────────────────────────────────────────────────────────

def process_cluster_contours(contours, labels, flowmap, alpha_mask, target_step=1.5, min_len=25, clusterindex = 0):
    global ysmoothmode, y1douglasepstargetmultdivider, yepsilon, ysmoothstrength, ytargetstep
    global ytargetptmultiplier, ySkipFirstSmooth, ySkipSecondSmooth, ySkipMergeResolve
    global yshort_tol, ymerge_tol
    
    """
    Für jede (Skelett-)Kontur:
      - vereinfachen + glätten
      - gleichmäßig abtasten
      - als Serie kleiner Quads in die Flowmap füllen (ohne selbstschneidende Polygone)
    """
    if len(contours) == 0:
        print(f"No contours in process_cluster_contours")
        return
    h, w = flowmap.shape[:2]
    print(f"CI[{str(clusterindex)}], process_cluster 1: {len(contours)} contours to work on...")
    if type(contours) == np.ndarray:
        curves = tuple((contour.reshape(-1, 2) if contour.ndim == 3 else contour).astype('float64') for contour in contours)
    elif type(contours) == list:
        curves = contours#[[pt[::-1] for pt in contour] for contour in contours] #curves = list(contour for contour in contours)
    for i, curve in enumerate(curves):
        draw_worm_ribbon(flowmap, labels, curves[i], alpha_mask = alpha_mask, color_from_tangent=True, clusterindex = clusterindex)

    SavePolys(outputfolder, file.name, file.format, h, w, curves, 1, output_polymapadder + "AF2_" + output_mergeresolve + "_" + str(clusterindex))
  
def new_cluster_image(
    img, 
    method="rgbxy", 
    spatial_weight=0.1, 
    superpixel_size=30, 
    compactness=10,
    quantile=0.1,
    n_samples=500
):
    """
    Cluster ein Bild mit erweiterten Methoden.
    
    Parameters
    ----------
    img : ndarray (H,W,3) oder (H,W,4)
        Eingabebild (RGB oder RGBA).
    method : str
        "rgbxy"    -> Clustering im erweiterten Feature-Space (R,G,B,x,y)
        "superpix" -> Cluster auf Superpixel-Basis
    spatial_weight : float
        Gewichtung für XY-Koordinaten im "rgbxy"-Modus.
    superpixel_size : int
        Durchschnittsgröße eines Superpixels (nur "superpix").
    compactness : float
        Kompaktheitsfaktor für SLIC (nur "superpix").
    quantile, n_samples : float, int
        Parameter für estimate_bandwidth.
    
    Returns
    -------
    labels : ndarray (H,W)
        Cluster-Labels pro Pixel.
    clustered : ndarray (H,W,3)
        Rekonstruiertes Bild aus Cluster-Zentren.
    """
    
    h, w = img.shape[:2]

    # Alpha-Kanal entfernen, falls vorhanden
    if img.shape[2] == 4:
        rgb = img[..., :3]
        alpha = img[..., 3]
        mask = alpha > 0
    else:
        rgb = img
        mask = np.ones((h, w), dtype=bool)

    if method == "rgbxy":
        # --- RGB + XY Feature-Space ---
        coords = np.indices((h, w)).transpose(1, 2, 0)  # (h,w,2)
        coords = coords.astype(np.float32) * spatial_weight
        features = np.concatenate([rgb.astype(np.float32), coords], axis=-1)

        flat = features[mask].reshape(-1, 5)

        bandwidth = estimate_bandwidth(flat, quantile=quantile, n_samples=n_samples)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(flat)

        labels_masked = ms.labels_
        centers = ms.cluster_centers_[:, :3].astype(np.uint8)  # nur RGB
        clustered_masked = centers[labels_masked]

        labels = -np.ones((h, w), dtype=int)
        clustered = np.zeros((h, w, 3), dtype=np.uint8)
        labels[mask] = labels_masked
        clustered[mask] = clustered_masked

        return labels, clustered

    elif method == "superpix":
        # --- Superpixel + Clustering ---
        superpixel_labels = segmentation.slic(
            rgb,
            n_segments=(h * w) // (superpixel_size**2),
            compactness=compactness,
            start_label=0
        )

        n_superpixels = superpixel_labels.max() + 1

        # Mittelwerte pro Superpixel
        superpix_features = np.zeros((n_superpixels, 3), dtype=np.float32)
        for sp in range(n_superpixels):
            mask_sp = (superpixel_labels == sp)
            if mask_sp.any():
                superpix_features[sp] = rgb[mask_sp].mean(axis=0)

        bandwidth = estimate_bandwidth(superpix_features, quantile=quantile, n_samples=n_samples)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(superpix_features)

        sp_centers = ms.cluster_centers_.astype(np.uint8)
        sp_labels = ms.labels_

        # Bild rekonstruieren
        clustered = np.zeros_like(rgb)
        labels = np.zeros((h, w), dtype=int)
        for sp in range(n_superpixels):
            clustered[superpixel_labels == sp] = sp_centers[sp_labels[sp]]
            labels[superpixel_labels == sp] = sp_labels[sp]

        return labels, clustered

    else:
        raise ValueError("Unknown method: choose 'rgbxy' or 'superpix'")

def cluster_image(img, method : Literal["meanshift", "kmeans", "slic"] = "meanshift"):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if method == "kmeans":
        flat = img.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=xn_clusters, n_init=10).fit(flat)
        labels = kmeans.labels_.reshape(h, w)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        clustered = centers[labels]
        return labels, clustered
    elif method == "meanshift":
        flat = img.reshape(-1, 3).astype(np.float32)
        xbandwidth = estimate_bandwidth(flat, quantile=xquantile, n_samples=xn_samples)
        ms = MeanShift(bandwidth=xbandwidth, bin_seeding=True).fit(flat)
        labels = ms.labels_.reshape(h, w)
        centers = ms.cluster_centers_.astype(np.uint8)
        clustered = centers[labels]
        return labels, clustered
    elif method == "slic":
        labels = segmentation.slic(img, n_segments=xn_clusters, compactness=10, start_label=0)
        clustered = color.label2rgb(labels, img, kind='avg').astype(np.uint8)
        return labels, clustered
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

def process_files(files, resetFolder = True):
    
    global outputfolder, file

    flowmaparr = []    
    for file in files:
        outputfolder = baseoutputfolder + "/" + file.name.replace(file.format,"") 
        if resetFolder: del_files_of_folder(outputfolder, sourcefileincludefilters, ["_snap_", "_clusters.png"], allowedimagetypes)
        better_print(f"{file.name}: Starting" , ShowTimeDiff=False, ResetTime=True, TimeGroup="processfiles_file")
        #outputfilebase = file.name[:-len(file.format)]
        Init_TimeGroup("processfiles_file_0")
        img = cv2.imread(file.fullname, cv2.IMREAD_UNCHANGED)
        if img is None:
            better_print(f"{file.name}: Image not found after: ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
            continue
        better_print(f"{file.name}: Image loaded after: ", ", starting generation of clusters...", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
        
        #Scaling factor for image size
        CurrImageBiggestSide = np.maximum(img.shape[0], img.shape[1])
        RefImagePropertySizeFactor = CurrImageBiggestSide / RefImageBiggestSide
        
        #Alpha mask + new img without alpha
        alpha_mask = None
        if img.shape[2] >= 4:
            alpha_mask = img[:,:,3].astype(bool)
            #alpha_mask = (alpha_mask < 254).astype(bool)
            img = img[:,:,:3]

        #Clustering
        ClusterImgPath = outputfolder + "/" + file.name.replace(file.format, "_" + output_clusteradder  + "_snap_" + file.format)
        if not LoadLastClusterimg or not (os.path.exists(ClusterImgPath)):
            if 1 == 1:
                labels, cluster_img = cluster_image(img, method=clustering_method) #new_cluster_image(img)
            else:
                labels, cluster_img = new_cluster_image(img, method="superpix", superpixel_size=20)
            SaveFlowmap(outputfolder, file.name, file.format, visualize_clusters(labels, img.shape), output_clusteradder)
            SaveFlowmap(outputfolder, file.name, file.format, labels, output_clusteradder + "_snap_")
            better_print(f"{file.name}: process_files 1: Clustered after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
        
        else:
            labels = cv2.imread(ClusterImgPath, cv2.IMREAD_UNCHANGED).astype(np.int64)
            better_print(f"{file.name}: process_files 1: Clusters loaded from {ClusterImgPath} ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
        
        if alpha_mask is not None:
            labels[~alpha_mask] = -1  # -1 = "kein Cluster"

        #labels = merge_small_clusters(labels, 200)
        
        ###MAYBE##
        #for cid in np.unique(labels):
        #    if cid == -1: continue
        #    mask = (labels == cid)
        #    if not np.count_nonzero(mask) >= 200 * RefImagePropertySizeFactor:
        #        labels[~mask] = -1
        ###MAYBE END##
        better_print(f"{file.name}: merged small labels after" , "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
        global currh, currw
        currh, currw = img.shape[:2]        
        # pro Cluster arbeiten
        jk = 0
        uniquelabels = [unlb for unlb in np.unique(labels) if not unlb == -1]
        better_print(f"{file.name}: process_files 2: working on {str(len(uniquelabels))} clusters after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
        Init_TimeGroup("processfiles_file_1")
        Init_TimeGroup("totaltime")
        fileflowmap = np.zeros((currh, currw, 4), np.uint8) * 127
        for i, cid in enumerate(uniquelabels):
            if cid == -1: continue  #alpha channel
            better_print(f"{file.name}: Cluster {str(i)} - Start after ", " - starting skeletizing", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_1")            
            
            jk += 1
            mask = (labels == cid)
            #"Too small Clusters dont matter"
            
            if not np.count_nonzero(mask) >= 500 * RefImagePropertySizeFactor:
                print(f"{file.name}: Cluster {str(i)} , process_files 2.2: {str(np.count_nonzero(mask))} points in cluster -> skipping... , border is {str(200 * RefImagePropertySizeFactor)}")
                continue
            else:
                print(f"{file.name}: Cluster {str(i)} , process_files 2.1: {str(np.count_nonzero(mask))} points in cluster to work on... , border is {str(200 * RefImagePropertySizeFactor)}")

            # Skeletonisieren & säubern
            skel = morphology.skeletonize(mask, method='zhang')
            # löcher schließen ->skeleton muss 1 breit sein, daher NEIN
            #just for testing!! actual skel gets generated in network from mask
            #skel = morphology.binary_closing(skel,)#, morphology.disk(1))            
            skel_img = np.full((currh, currw, 4), (0,0,0,0), dtype=np.uint8)
            skel_img[skel] = (255,255,255,255)
            SaveFlowmap(outputfolder, file.name, file.format, skel_img, output_skeletonadder + f"_label({str(jk)})")
            better_print(f"{file.name}: Cluster {str(i)} , Skeletized for ", ", starting generation of networking...", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
            #netzwerke generieren
            networks = network(mask)
            networks = networks.connect_split_process_network_to_networks()
            
            better_print(f"{file.name}: Cluster {str(i)} , Networked for ", f", got {len(networks)} seperate networks out of cluster. Starting generation of Flowmap...", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
            #fileflowmap = np.zeros((currh, currw, 4), np.uint8) * 127
            for af, networ in enumerate(networks):
                flowmap = np.zeros((currh, currw, 4), np.uint8) * 127  # 127 = kein Flow
                pointpaths = []
                if len(networ.network) > 0:
                    for j in networ.network.keys():
                        for pp in list(j.upstreampointandpaths.values()) + list(j.downstreampointandpaths.values()):
                            if not pp in pointpaths: pointpaths.append(pp)
                else:
                    pointpaths.extend(networ.singlepaths)
                if DrawMode == "Old":
                    ppathlist = [[pt[::-1] for pt in ppath.points] for ppath in pointpaths]
                    process_cluster_contours(ppathlist, labels, flowmap, alpha_mask=networ.clustermask, target_step=0.5, min_len=15, clusterindex=jk)
                    fileflowmap[flowmap != (0, 0, 0, 0)] = flowmap[flowmap != (0, 0, 0, 0)]
                    FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, flowmap, output_flowmapadder + f"_cluster_{str(jk)}_network{str(af)}")
                elif DrawMode == "0.5":
                    a = generate_flowmap(networ, 1000, 50) #flowmap_from_junctions([junc for junc in networ.network], networ, plot = 'both')
                    fileflowmap[a != (0, 0, 0, 0)] = a[a != (0, 0, 0, 0)]
                    better_print(f"{file.name}: Cluster {str(i)} , generated flowmap for ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="networkgenerated")
                    FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, a, output_flowmapadder + f"_cluster_{str(jk)}_network{af}_", "networkgenerated")
            
            # Konturen zur sicherheit mit maske abgleichen und außenliegende entfernen
            #contours = filter_contours_by_mask(contours, alpha_mask)
            #print(f"processfiles 2.5: {str(len(contours))} AF filter_contours_by_mask")
            #SavePolys(outputfolder, file.name, file.format, currh, currw, contours, 1, output_skeletonadder + "_contours_maskfilter_" + str(jk))
            
            
            #process_cluster_contours(contours, labels, flowmap, alpha_mask=mask, target_step=0.5, min_len=15, clusterindex=jk)
            #
            #FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, flowmap, output_flowmapadder + f"_cluster_{str(jk)}")
            #better_print(f"{file.name}: Flowmap generated after ", f", Cluster {str(jk)}: ({FlowmapFilePath})", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file_0")
            #flowmaparr.append([file, mask, flowmap, jk])
            #better_print(f"{file.name}: Ending after:", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file")
    
            FlowmapFilePath = SaveFlowmap(outputfolder, file.name, file.format, fileflowmap, output_flowmapadder + f"_cluster_{str(jk)}_networks_total_", "networkgenerated")
            flowmaparr.append(fileflowmap)
        better_print(f"##FILE ({file.name}) FINISH## after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="processfiles_file")
    better_print(f"##TOTAL FINISH## after ", "", ShowTimeDiff=True, ResetTime=True, TimeGroup="totaltime")            
            
    #import flowmap_vektor_length_unionizer as fvlu
    #flowmaps = fvlu.process_files(flowmaparr)
    #import os
    #os.system('shutdown /p /f') 

if __name__ == "__main__":
    process_files(files)

#-> cute parameter mode: Literal["angle", "neighbourcount", "neighbourdir", "neighbourdirold"] = "neighbourcount"