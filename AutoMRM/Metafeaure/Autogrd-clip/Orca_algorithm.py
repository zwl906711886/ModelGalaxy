import subprocess
def createStructureFiles(path_from, path_to):
    """
        This function takes as an input the .in file, which created by the co-occurence computation and splits the lines, in order
        to fit the file to the format that required by Orca algorithm.

        Parameters
        ----------
        path_from : str
            path to the location of the .in file, which created by the co-occurence computation
        path_to : str
             path to the location of the output file, in which the lines are separated.
        """
    fh = open(path_from, "r")
    lines = fh.readlines()
    wri = open(path_to, "w")
    wri.writelines(lines)

def OperateOrca(path_from, path_to_ndump2):
    """
        This function call to the orca algorithm. The workspace director must contain the orca.exe file.
        The implementation of Orca is availiable at http://www.biolab.si/supp/orca/.

        Parameters
        ----------
        path_from : str
            path to the location of the file that contain the structure of the graph, the filename extension should be .in
        path_to_ndump2 : str
             path to the location of the file, where the results of the orca algorithm will be saved. The filename extenstion of the
             output file will be .ndump2
        """
    subprocess.call("./orca 4 "+ path_from + ' '+ path_to_ndump2+".ndump2",shell=True)


path_from1 = "output24/in/dataset.in"
path_to1 = "output24/in/process/dataset.in"
path_to2 = "output32/dataset"
createStructureFiles(path_from1, path_to1)
OperateOrca(path_to1, path_to2)

