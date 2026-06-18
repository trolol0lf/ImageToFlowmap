if __name__ == "__main__":
    import networker_5 as gpt5 ##Current main script
    gpt5.process_files(gpt5.files)
    import sys
    sys.exit()

import re
import os
from PySide6.QtWidgets import QFileDialog
from click import Option
from param import Filename
from sklearn.cross_decomposition import PLSCanonical
from sympy import EX, false
import cv2
import numpy as np
import random as rnd
from typing import Literal, Tuple, Set, Optional, List, Dict

from tkinter import Tk, filedialog

def select_folder_ui(title="Bitte Ordner auswählen", initialdir = None):
    root = Tk()
    root.withdraw()  # kein extra Fenster
    if initialdir and os.path.exists(initialdir):
        folder = filedialog.askdirectory(title=title, initialdir=initialdir)
    else:
        folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

def select_file_ui(title="Bitte Datei auswählen", initialdir=None, filetypes=(("Alle Dateien","*.*"),)):
    root = Tk(); root.withdraw()
    if initialdir and os.path.exists(initialdir):
        path = filedialog.askopenfilename(title=title, initialdir=initialdir, filetypes=filetypes)
    else:
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

from tables import Unknown

class AccessManager():
    def __init__(self):
        self.AccessGroups : Dict[str , List[str]] = {}

    def can_access(self, AccessGroups : List[str]):
        CanContinue = 1
        for GroupMembers in self.AccessGroups.values():
            CanContinue = 0
            rmag = []
            for ag in AccessGroups:
                if CanContinue == 0: CanContinue = -1
                if ag in GroupMembers:
                    CanContinue = 1
                    rmag.append(ag) ##use keys multiple times?
                    break
            
            for ag in rmag:
                AccessGroups.remove(ag)
            if CanContinue <= 0: break
        return CanContinue == 1
    
    def set_AccessGroup(self, AccessGroup : Dict[str , List[str]]):
        for Groupname, Group in AccessGroup.items():
            self.AccessGroups[Groupname] = Group

    def add_AccessGroupmember(self, AccessGroup : Dict[str , List[str]]):
        for Groupname, Group in AccessGroup.items():
            if Groupname in self.AccessGroups.keys():
                for Mem in Group:
                    if not Mem in self.AccessGroups[Groupname]:
                        self.AccessGroups[Groupname].append(Mem)
            else:
                self.AccessGroups[Groupname] = Group

    def remove_AccessGroup(self, AccessGroup : str):
        if AccessGroup in self.AccessGroups.keys():
            return self.AccessGroups.pop(AccessGroup)
        return None
    
    def remove_AccessGroupMember(self, AccessGroup : str, Member : str):
        if AccessGroup in self.AccessGroups.keys():
            if Member in self.AccessGroups[AccessGroup]:
                return self.AccessGroups[AccessGroup].pop(Member)
        return None
    
class xfilelist:
    def __init__(self, initxfiles = [], initfolder = ""):
        self.filelist = []
        for item in initxfiles:
            self.append(item)
        if os.path.exists(initfolder):
            newfiles = self.get_files_of_folder(initfolder)
            for item in newfiles:
                self.append(item)

    def get_files_of_folder(folder_path, includefilters = [], excludefilters = [], allowedtypes = [], walksubdirs = False):
        # Code to get all files in a folder
        filelist = []
        if os.path.exists(folder_path):
            currdir = None
            for root, dirs, files in os.walk(folder_path):                
                for file in files:
                    if (len(includefilters) == 0 or name_contains_arrayelem(file, includefilters)) and not name_contains_arrayelem(file, excludefilters):
                        allowedtype = False
                        for allotype in allowedtypes:
                            if file[-len(allotype):].lower() == allotype:
                                allowedtype = True
                                break
                        if allowedtype or len(allowedtypes) == 0:
                            filelist.append(os.path.join(root, file))
                if not walksubdirs:
                    break
        return filelist

    def __getitem__(self, index):
        return str(self.filelist[index])
    def GetRealItem(self, index):
        return self.filelist[index]
    def __len__(self):
        return len(self.filelist)
    def append(self, item):
        if type(item) == xfile:
            self.filelist.append(item)
        elif type(item) == str:
            newfile = xfile(item)
            if newfile:
                self.filelist.append(newfile)
        else:
            raise Exception("No valid item type for new xfile of xfilelist!")
        
    def __iter__(self):
        return iter(self.filelist)
    
    def __next__(self):
        return next(self.filelist)
    
class xfile:
    def __init__(self, root, name = ""):  
        root = root.replace("\\", "/")
        if os.path.exists(os.path.join(root,name)) and not name == "":      
            self.root = root
            self.name = name
            self.fullname = os.path.join(root, name)
            self.format = re.search(r"\.[a-zA-Z]{1,4}$", name).group()
        elif os.path.isfile(root) and name == "":
            self.fullname = root
            self.name = os.path.basename(root)
            self.root = root.replace(self.name, "")            
            self.format = re.search(r"\.[a-zA-Z]{1,4}$", self.name).group()
        else:
            return False
    def __repr__(self):
        return self.name

def select_folder():
    filename = QFileDialog.getExistingDirectory(caption="Select directory", options=QFileDialog.Option.DontUseNativeDialog)
    return filename   #Returns None if invalid?
    
def name_startswith_arrayelem(name, arr = []):
    for i in arr:
        if name.startswith(i):
            return True
    return False
def name_endswith_arrayelem(name, arr = []):
    for i in arr:
        if name.endswith(i):
            return True
    return False

def name_contains_arrayelem(name, arr = []):
    for i in arr:
        if name.find(i) != -1:
            return True
    return False

def get_files_of_folder(folder_path, includefilters = [], excludefilters = [], allowedtypes = [], walksubdirs = False):
    # Code to get all files in a folder
    filelist = xfilelist()
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if (len(includefilters) == 0 or name_contains_arrayelem(file, includefilters)) and not name_contains_arrayelem(file, excludefilters):
                    allowedtype = False
                    for allotype in allowedtypes:
                        if file[-len(allotype):].lower() == allotype:
                            allowedtype = True
                            break
                    if allowedtype or len(allowedtypes) == 0:
                        filelist.append(os.path.join(root, file))
            if not walksubdirs: break
    return filelist

def del_files_of_folder(folder_path, includefilters = [], excludefilters = [], allowedtypes = []):
    # Code to get all files in a folder
    removelist = []
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if (len(includefilters) == 0 or name_contains_arrayelem(file, includefilters)) and not name_contains_arrayelem(file, excludefilters):
                    allowedtype = False
                    for allotype in allowedtypes:
                        if file[-len(allotype):].lower() == allotype:
                            allowedtype = True
                            break
                    if allowedtype or len(allowedtypes) == 0:
                        removelist.append(os.path.join(root, file))
        for rmpath in removelist:
            os.remove(rmpath)
    return removelist

def Recursive_Get_AllPointsInListSetTuples(Polys, pts = []):
    #Gets either if its a point list or a list of pointlists
    if isinstance(Polys, (list, set, tuple)):
        for p in Polys:
            if isinstance(p, (int, int)):
                pts.append(p)
            elif isinstance(p, (list, set, tuple)):
                Recursive_Get_AllPointsInListSetTuples(p, pts)
        return pts
    else:
        raise Exception(f"Unknown list type for Get_PolyListType's Polys: {str(Polys)}")

def Get_PolyListType(Polys):
    #Gets either if its a point list or a list of pointlists
    if isinstance(Polys, (list, set, tuple)):
        plisttype = "unknown" #either it is an pointlist or a list of pointlists
        for p in Polys:
            if isinstance(p, (list, set, tuple)):
                for subp in p:
                    if isinstance(subp, (int, np.int64, np.int32)):
                        if plisttype == "unknown":
                            plisttype = "pointlist"
                        elif not plisttype == "pointlist":
                            return "unknown"
                    elif isinstance(p, (list, set, tuple)):
                        if plisttype == "unknown":
                            for subsubp in subp:
                                if isinstance(subsubp, int):
                                    if plisttype == "unknown":
                                        plisttype = "listofpointlists"
                                    elif not plisttype == "listofpointlists":
                                        return "unknown"
            else:
                raise Exception(f"Unknown list element type for Get_PolyListType's p: {str(p)}")
    elif isinstance(Polys, np.ndarray):
        return "npndarr"
    elif isinstance(Polys, dict):
        return "dict"
    else:
        raise Exception(f"Unknown list type for Get_PolyListType's Polys: {str(Polys)}")
    return plisttype

FlowMapCounters = {}

def Get_FlowMapCounterAdder(FlowmapCounterGroup = ""):
    CounterAdder = ""
    if FlowmapCounterGroup != "":
        global FlowMapCounters
        if FlowmapCounterGroup in FlowMapCounters.keys():
            FlowMapCounters[FlowmapCounterGroup] += 1
        else:
            FlowMapCounters[FlowmapCounterGroup] = 0
        CounterAdder = str(FlowMapCounters[FlowmapCounterGroup]) + "_"
    return CounterAdder
                                                                                                                #random combination for standard
def SavePolys(BasePath, FileName, FileFormat, h, w, Polys, scale = 1, PostFix = "debug", FlowmapCounterGroup = "-&--ximzz--&-", backgroundcolor = (0,0,0,0)):
    h_hr, w_hr = h * scale, w * scale
    PolyLineImg = np.full((h_hr, w_hr, 4), backgroundcolor, dtype=np.uint8)
    thickness = 1
    plisttype = Get_PolyListType(Polys=Polys)
    pts = None
    if plisttype == "unknown":
        print(f"Unknown Point list type, can not save: {PostFix} (-{FileName})")
        return
    elif plisttype == "listofpointlists" or plisttype == "dict" or plisttype == "npndarr":
        for poly in Polys:
            if type(poly) == dict:
                pts = (poly['curve'] * scale).astype(np.int32)
            elif type(poly) == np.ndarray:
                pts = (poly * scale).astype(np.int32)
            elif type(poly) == list or type(poly) == set:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(PolyLineImg, [pts], isClosed=False, color=(rnd.randint(0,255), rnd.randint(0,255), rnd.randint(0,255)), thickness=thickness)
    elif plisttype ==  "pointlist":
        pts = np.array(Polys, dtype=np.int32)
        cv2.polylines(PolyLineImg, [pts], isClosed=False, color=(rnd.randint(0,255), rnd.randint(0,255), rnd.randint(0,255)), thickness=thickness)
        
    #Outputting polylinemap
    SaveFlowmap(BasePath, FileName, FileFormat, PolyLineImg, PostFix, FlowmapCounterGroup)

def encode_vec_dtype(vx, vy, bitcount : int):
    """
    Erwartet Komponenten im Bereich [-1,1].
    Mappt auf [0,255] per 0.5*(v)+0.5.
    """
    dtype, max = get_datatype(bitcount)
    
    def enc(c):
        return np.clip((0.5 * c + 0.5) * float(max), 0, int(max)).astype(dtype)
    return enc(vx), enc(vy)

def get_datatype(bitcount):
    match bitcount:
        case 8:
            return np.uint8, (2**8)-1
        case 16:
            return np.uint16, (2**16)-1
        case 32:
            return np.uint32, (2**32)-1
        case 64:
            return np.uint64, (2**64)-1
    raise Exception(f"Unknown bitcount in get_datatype: {bitcount}")
def save_vec_field_as_flowmap(vec_field, strength, H, W, cy, cx, usestrength = True, bitcount : Literal[8, 16, 32, 64] = 16):
    """
    vec_field: (M,2)
    strength: (M,)
    H,W: shape
    cy,cx: pixel indices
    """
    dtype, max = get_datatype(bitcount)
    if usestrength:
        vy, vx = vec_field[:,0] * strength, vec_field[:,1] * strength
    else:
        vy, vx = vec_field[:,0], vec_field[:,1]
    enc_x, enc_ny = encode_vec_dtype(vx, -vy, bitcount=bitcount)

    flow = np.zeros((H,W,4), dtype=dtype)
    flow[cy, cx, 3] = max
    flow[cy, cx, 1] = enc_ny
    flow[cy, cx, 2] = enc_x
    return flow

def ensure_save_flowmap_format(FlowMap, AddAlpha = False, Normalize = False, bitcount : Literal[8, 16, 32, 64] = 16, Resize = True):
    dtype, max = get_datatype(bitcount)
    ActualMap = FlowMap.copy()
    if ActualMap.dtype == np.dtype('bool'):
        if Resize:
            ActualMap = ActualMap.astype(dtype) * max
    elif (ActualMap.dtype == np.dtype('float64') or ActualMap.dtype == np.dtype('float32')):
        if Normalize and (ActualMap.max() > 1.0 or ActualMap.min() < 0.0):
            ActualMap += ActualMap.min()
            ActualMap /= ActualMap.max() * max
        if Resize:
            ActualMap = ActualMap * max
    elif ActualMap.dtype != dtype:
        if dtype == np.uint8 and ActualMap.dtype == np.uint16:
            ActualMap = (ActualMap.astype(np.float64) / (2**16-1) * (2**8-1)).astype(dtype)
        elif dtype == np.uint16 and ActualMap.dtype == np.uint8:
            ActualMap = (ActualMap.astype(np.float64) / (2**8-1) * (2**16-1)).astype(dtype)
    ActualMap = ActualMap.astype(dtype)
    # Falls Alpha erwünscht -> auf BGRA erweitern
    if len(ActualMap.shape) <= 2:  # Graustufen
        ActualMap = cv2.cvtColor(ActualMap, cv2.COLOR_GRAY2BGR)
    elif  ActualMap.shape[2] == 2:
        tmap = np.zeros((ActualMap.shape[0], ActualMap.shape[1], 3), dtype=dtype)
        tmap[..., 1:] = ActualMap[..., :]
        ActualMap = tmap
    if AddAlpha:
        if ActualMap.shape[2] == 3:  # BGR
            ActualMap = cv2.cvtColor(ActualMap, cv2.COLOR_BGR2BGRA)
        # Alpha setzen: 0 wo alles 0, sonst 255
        ActualMap[..., 3] = max
        ActualMap[..., 3] = np.where(np.all(ActualMap[..., :3] == 0, axis=-1), 0, max)# | np.where(ActualMap[..., 1] == 0.0, 0, max) | np.where(ActualMap[..., 0] == 0.0, 0, max)
    return ActualMap

def SaveFlowmap(BasePath, FileName, FileFormat, Flowmap,
                PostFix="debug", FlowmapCounterGroup="", AddAlpha=False, Normalize = False):
    CounterAdder = Get_FlowMapCounterAdder(FlowmapCounterGroup)
    if not isinstance(Flowmap, np.ndarray):
        return False

    ActualMap = ensure_save_flowmap_format(FlowMap=Flowmap, AddAlpha=AddAlpha, Normalize = Normalize)
    
    replacer = "_" + PostFix + FileFormat if not PostFix == "" else FileFormat
    betterfilename = FileName.replace(FileFormat, replacer) if FileFormat in FileName else FileName + replacer
    targetfilepath = os.path.join(
        BasePath,
        CounterAdder + betterfilename
    )
    if not os.path.exists(os.path.dirname(targetfilepath)):
        os.makedirs(os.path.dirname(targetfilepath))
    if os.path.exists(targetfilepath):
        os.remove(targetfilepath)

    cv2.imwrite(targetfilepath, ActualMap)
    return targetfilepath

def olSaveFlowmap(BasePath, FileName, FileFormat, Flowmap, PostFix = "debug", FlowmapCounterGroup = "", AddAlpha = False):
    CounterAdder = Get_FlowMapCounterAdder(FlowmapCounterGroup)
    if not isinstance(Flowmap, np.ndarray): return False
    ActualMap = Flowmap
    if ActualMap.dtype == np.dtype('bool'):
        ActualMap = ActualMap.astype(np.uint8) * 255
    targetfilepath = os.path.join(BasePath, CounterAdder + FileName.replace(FileFormat, "_" + PostFix + FileFormat))
    if not os.path.exists(os.path.dirname(targetfilepath)):
        os.makedirs(os.path.dirname(targetfilepath))
    if os.path.exists(targetfilepath):
        os.remove(targetfilepath)
    cv2.imwrite(targetfilepath, ActualMap)
    return targetfilepath

def PointsToFlowmap(points, h, w, scale=1, point_color=None, radius=2, backgroundcolor = (0,0,0,0), bitsize : Literal[8, 16] = 16):
    """
    Erzeugt eine Flowmap (RGB-Bild), die Punkte darstellt.
    
    Args:
        points      : Liste oder Menge von (y,x)-Koordinaten
        h, w        : Höhe und Breite des Zielbildes
        scale       : Skalierungsfaktor
        point_color : Optional fester RGB-Farbwert (z.B. (0,0,255)).
                      Falls None, wird für jeden Punkt eine Zufallsfarbe genommen.
        radius      : Radius der Punkte (Pixel)
    
    Returns:
        Flowmap : np.ndarray (HxWx3, uint8)
    """
    dt, maxbit = get_datatype(bitsize)
    h_hr, w_hr = int(h * scale), int(w * scale)
    Flowmap = np.full((h_hr, w_hr, 4), backgroundcolor, dtype=dt)

    for (y, x) in points:
        cy, cx = int(y * scale), int(x * scale)
        if point_color is None:
            color = (rnd.randint(maxbit/5,maxbit), rnd.randint(maxbit/5,maxbit), rnd.randint(maxbit/5,maxbit), maxbit)
        else:
            color = point_color
        cv2.circle(Flowmap, (cx, cy), radius=radius, color=color, thickness=-1)

    return Flowmap

def cache_result(path, compute_fn, *args, **kwargs):
    """
    Lädt ein Ergebnis aus `path`, wenn vorhanden, 
    sonst berechnet es compute_fn(*args, **kwargs) und speichert es.
    """
    if os.path.exists(path):
        print(f"[Cache] Lade {path}")
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    print(f"[Cache] Erzeuge {path}")
    result = compute_fn(*args, **kwargs)
    cv2.imwrite(path, result)
    return result

from datetime import datetime
TimeGroups = {}
StartTime = datetime.now()
LastTime = StartTime

def getTimeString(Time1 = datetime.now(), Time2 = datetime.now(), TimeMode = "Seconds"):
    match TimeMode:
        case "Hours":
            return str(round((Time1 - Time2).total_seconds() / 3600, 2)) + " hours"
        case "Minutes":
            return str(round((Time1 - Time2).total_seconds() / 60, 2)) + " minutes"
        case "Seconds":
            return str(round((Time1 - Time2).total_seconds(), 2)) + " seconds"
    raise Exception(f"No valid TimeMode: {TimeMode}")

def Update_TimeGroup_Get_Times(TimeGroup, ResetTime = True):
    global TimeGroups
    locLastTime = datetime.now()
    locNewTime = locLastTime
    OutlocLastTime = locLastTime

    if TimeGroup in TimeGroups.keys():
        locLastTime = TimeGroups[TimeGroup][1]
        OutlocLastTime = locLastTime    #We save the last time and return it. The value gets updated in the dictionary if demanded
        locNewTime = datetime.now()
        if ResetTime:
            locLastTime = locNewTime
    
    TimeGroups[TimeGroup] = (locNewTime, locLastTime)
    return locNewTime, OutlocLastTime

def Init_TimeGroup(TimeGroup):
    global TimeGroups    
    TimeGroups[TimeGroup] = (datetime.now(), datetime.now())

def better_print(PrePrintStr = "", PostPrintStr = "", ShowTimeDiff = True, ResetTime = True, TimeGroup = "t13-.133"):
    ThisNewTime, ThisLastTime = Update_TimeGroup_Get_Times(TimeGroup, ResetTime)
    if ShowTimeDiff:
        NewTime = datetime.now()
        print(f"{PrePrintStr}{getTimeString(ThisNewTime, ThisLastTime)}{PostPrintStr}")
        if ResetTime: LastTime = NewTime
    else:
        print(PrePrintStr + PostPrintStr)