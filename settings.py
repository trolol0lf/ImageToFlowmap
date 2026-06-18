if __name__ == "__main__":
    import networker_5 as nw5 ##Current main script
    #gpt5_rework_3 best so far   
    nw5.process_files(nw5.files)
    import sys
    sys.exit()
    
from FileHandler import *
import sys
from typing import Literal, Union, Dict, Optional

class cwrap():
    def __init__(self):
        pass

processtypes = Literal["Clustering", "NetworkGeneration", "NetworkSplitting", "Junctioning", "Pathing", "FlowmappingAll", "FlowmappingSingle"]

standardprocesssignals = Literal["Process", "QTimer_Update", "QTimer_Terminate"]
##Cluster GPT5
clustering_methods = Literal["meanshift", "kmeans", "slic"]
clustering_method : clustering_methods = "meanshift"

clustermethodparameters : Dict[str , Dict[str , Union[str, int, float]]] = \
                                {
                                    "meanshift" : {
                                                    "Quantile"  : 0.2,
                                                    "Samples"   : 500
                                                   },

                                    "kmeans"    :   {
                                                    "Target Clustercount"   : 4,
                                                    "n init"                : 10,
                                                    },

                                    "slic"      :   {
                                                    "Target Clustercount"    : 4,
                                                    "Compactness"           : 10
                                                    }
                                }
#kmeans/slic
xn_clusters = 4                # Nur für kmeans/slic
#meanshift
xquantile = 0.2                # MeanShift
xn_samples = 500               # MeanShift

##Cluster End

##RefImage biggest side size. This will scale other properties if the input image biggest side size varies
RefImageBiggestSide = 512
CurrImageBiggestSide = 512
##The corresponding current size factor
RefImagePropertySizeFactor = 1
##RefContourLength -> Will influence the point count for smoothing
RefContourLength = 5

#Skeleton Generator
SkeletonType : Literal["lee", "zhang", "weird_splitted_skeleton"] = "weird_splitted_skeleton"

##SimplifyAndSmooth
yksize = 8
ySkipMergeResolve = True
yshort_tol = 6
ymerge_tol = 3
#1
ysmoothmode = ["douglas-pecker", "simplifyandsmooth"][0]
y1epsilon = 3.2
y1douglasepstargetmultdivider = 10
y1smoothstrength = 20
y1targetstep = 2.5
y1targetptmultiplier = 15
ySkipFirstSmooth = True
#2
y2epsilon = 0.5
y2smoothstrength = 5
y2targetstep = 0.5
y2targetptmultiplier = 15
ySkipSecondSmooth = True
##SimplifyAndSmooth


##Boundary finder
maxboundarymarchdist = 250
initiallastboundarymarchdist = -1
boundarytolerance = 0.1
lastboundarymarchdist = initiallastboundarymarchdist
##Boundary finder end

##Misc
output_clusteradder = "clusters"
output_skeletonadder = "skeleton"
output_flowmapadder = "flowmap"
output_polymapadder = "poly_"
output_unionizedadder = "unionized"
output_mergeresolve = "mergeresolve"
output_contouradder = "contour"
# === Parameter for Unionizer ===

fvlu_blurruns = 2
fvlu_threshold = 0               # Schwellenwert für Vektorlänge
fvlu_tolerance = 5
##Misc End

AccessMngr = AccessManager()



DebugMode = True
DrawMode : Literal["Old", "0.5", "gpt08-25"] = "gpt08-25"
LoadLastClusterimg = True
sourcefolder = r"C:/Users/Lolf/Desktop/curstate goblic/PhoenixGraphics/PhoenixLogo/2048Skaled/"

sourcefileincludefilters = [["nebel_s",                     #0
                            "TestAlpha",                    #1
                            "Phoenix_vector_seperateRed",   #2
                            "Phoenix_Vector_all",           #3
                            "part_red_single",              #4
                            "phoenix_all_512",              #5
                            "phoenix_all_multjunc_3",       #6
                            "phoenix_all_2048_alpha",       #7
                            "Phoenix_Vector_4096",          #8
                            "Phoenix_Vector_8192_bicolor",  #9
                            "4096_lower_wirbel"             #10
                            ][8]]

sourcefileexcludefilters = [
                            "_" + output_flowmapadder,
                            "_" + output_skeletonadder,
                            "_" + output_clusteradder,
                            "_" + output_polymapadder,
                            "_" + output_mergeresolve,
                            "_" + output_unionizedadder,
                            "_flip",
                            "_binary"
                            ]

allowedimagetypes = [".png",".jpg",".jpeg",".bmp"]
baseoutputfolder = os.path.join(sourcefolder, clustering_method)
outputfolder = baseoutputfolder
file = None

ufinenum = 5000
radialspacestartstop = 150
radialspacenum = 300
gaussian_sigma = 0.9
gaussian_radius = 3

if sourcefolder == "" or not os.path.exists(sourcefolder):
    print("No sourcefolder : " + sourcefolder)
    sys.exit(0)

files = get_files_of_folder(sourcefolder, sourcefileincludefilters, sourcefileexcludefilters, allowedimagetypes)

if len(files) == 0:
    print("No files in: " + sourcefolder + " ; for includefilter: '" + sourcefileincludefilters + "' and excludefilter '" +  + "'")
    sys.exit(0)