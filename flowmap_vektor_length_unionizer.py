

import array
import cv2, os
import numpy as np
from FileHandler import *
from settings import *

def masked_median_blur(flowmap, mask, ksize=5, ignore_color=(0,127,127)):
    mask = (mask>0).astype(np.uint8)
    pad = ksize // 2
    h, w = flowmap.shape[:2]
    out = flowmap.copy()
    dttype = flowmap.dtype
    # Padding, damit Randpixel funktionieren
    padded_img = cv2.copyMakeBorder(flowmap, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    padded_mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    channels = flowmap.shape[2] if flowmap.ndim == 3 else 1
    cy, cx = np.where(mask)

    for y,x in zip(cy, cx):        
        if mask[y, x] == 0:
            continue  # nur maskierte Bereiche filtern

        # Ausschnitt
        roi = padded_img[y:y+ksize, x:x+ksize]
        roi_mask = padded_mask[y:y+ksize, x:x+ksize]

        # Nur Nachbarn innerhalb Maske UND nicht ignore_color
        if channels == 3:
            # Pixel, die nicht ignore_color sind
            ignore_mask = np.all(roi == ignore_color, axis=-1)  
            valid_mask = (roi_mask > 0) & ~ignore_mask
        else:
            # Kein ignore_color bei 2D-Flows
            valid_mask = roi_mask > 0

        valid_pixels = roi[valid_mask]

        if len(valid_pixels) > 0:
            # Median pro Kanal
            out[y, x] = np.median(valid_pixels, axis=0).astype(dttype)

    return out

def process_files(files):
    
    global fvlu_threshold, fvlu_tolerance, fvlu_blurruns
    results = []
    
    totalimages = {}

    for file in files:
        alpha_mask = None
        clusterindex = 0
        if type(file) == xfile:
            output_path = file.name.replace(file.format, "_" + output_unionizedadder + file.format)
            # === Bild laden ===
            # Annahme: R = x-Komponente, G = y-Komponente, B wird ignoriert
            img = cv2.imread(file.fullname, cv2.IMREAD_UNCHANGED).astype(np.float64)
        #file, mask, flowmap if from gpt5_rework_3
        elif type(files) == list:
            img = file[2].astype(np.float64)
            alpha_mask = file[1]
            clusterindex = file[3]            
            if not file in totalimages.keys():
                h, w = img.shape[:2]  
                totalimages[file] = np.ones((h, w, 3), np.uint8) * 127  # 127 = kein Flow
                totalimages[file][..., 0] = 0
            
        if img is None:
            print(FileNotFoundError(f"Bild nicht gefunden: {img}"))
            better_print(f"Bild nicht gefunden ", ": " + file.name, True, False)
            continue
        #Alpha mask + new img without alpha
        
        if img.shape[2] >= 4:
            alpha_mask = img[:,:,3].astype(bool)
            img = img[:,:,:3]

        # R = x, G = y
        x = img[:, :, 2]  # R-Kanal
        y = img[:, :, 1]  # G-Kanal

        # Zentrieren (falls 128 = 0 ist)
        x -= 127.0
        y -= 127.0

        # Vektorlänge berechnen
        magnitude = np.sqrt(x**2 + y**2)

        # Maske: wo der Vektor länger als Threshold ist
        mask = magnitude > fvlu_threshold + fvlu_tolerance

        # Normalisieren (nur dort wo nötig)
        x[mask] = x[mask] / magnitude[mask] * 127.5
        y[mask] = y[mask] / magnitude[mask] * 127.5

        # Zurückschieben in [0,255] Bereich
        x += 128.0
        y += 128.0

        # Neue Kanäle zusammenbauen
        result = img.copy()
        result[:, :, 2] = np.clip(x, 0, 255)
        result[:, :, 1] = np.clip(y, 0, 255)

        # In uint8 zurückwandeln und speichern
        result = result.astype(np.uint8)
        NewFile = SaveFlowmap(file.root, file.name, file.format, result, output_unionizedadder)
        flowmap = img
        for i in range(fvlu_blurruns):
            flowmap = masked_median_blur(flowmap=flowmap, mask = mask, ksize=yksize)
        if type(file) == list: totalimages[file][alpha_mask] = flowmap[alpha_mask]
        SaveFlowmap(outputfolder, file.name, file.format, flowmap, output_polymapadder + str(clusterindex) + "_blurred")
        better_print(f"Flowmap blurred ", ": " + file.name + f" Cluster {str(clusterindex)}: ({yksize})")

        results.append(result)
        print(f"Flowmap von {file.name} erfolgreich normalisiert und gespeichert.")

    for keyval in totalimages.keys():
        SaveFlowmap(outputfolder, keyval.name, keyval.format, totalimages[keyval], output_polymapadder + str(clusterindex) + "_blurred_combined" )

    return results

if __name__ == "__main__":
    inputfolder = "C:/Users/Lolf/Desktop/curstate goblic/PhoenixGraphics/PhoenixLogo/2048Skaled/meanshift/"
    input_paths = ["flowmap"]     # Pfad zum Input-Bild
    files = get_files_of_folder(inputfolder, input_paths, [output_unionizedadder], [".png", ".jpg"])
    process_files(files)
