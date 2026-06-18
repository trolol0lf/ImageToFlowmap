import logging, os, sys

# Setze den Log-Level für die matplotlib-Bibliothek auf INFO
logging.getLogger('matplotlib').setLevel(logging.INFO)

# Setze den Log-Level für die fontManager-Bibliothek auf INFO
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

from typing import Literal, Dict

logger = logging.getLogger()

leveltype = Literal["debug", "info", "warning", "error"]
Debugging = True
if Debugging: print("running logger.py")
leveltypes : Dict[leveltype, int] = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR
        }

logger.setLevel(0)
if os.path.exists(os.path.join(os.path.dirname(sys.argv[0]), "Log.txt")):
    os.remove(os.path.join(os.path.dirname(sys.argv[0]), "Log.txt"))

file_handler = logging.FileHandler(os.path.join(os.path.dirname(sys.argv[0]), "Log.txt"), encoding="utf-8")
# Setze den Formatter für den FileHandler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
# Füge den FileHandler zum Logger hinzu
logger.addHandler(file_handler)

def addlog(level : leveltype | int, text : str):
    global Debugging, leveltypes
    if type(level) == str:
        rlevel = leveltypes[level]
    else:
        rlevel = level
    if Debugging: logger.log(rlevel, text)