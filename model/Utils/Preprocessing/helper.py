import networkx as nx
import os

def extract_apk_name(file_path: str):
    """
    Extract the name of the APK file from the path of the file.
    :param file_path: the path of the file
    :return: the name of the APK file
    """
    try:
        apk_name = os.path.basename(file_path).split('.')[0]
        return apk_name
    except:
        raise Exception("The file path is not correct: %s" % file_path)

def read_callgraph(file_path: str):
    """
    Read the callgraph from a file.
    :param file_path: the path of the file
    :return: a networkx graph
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("The file does not exist: %s" % file_path)
    CG = nx.read_gml(file_path)

    return CG