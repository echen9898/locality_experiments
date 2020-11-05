
import os

def create_path(path):
    """ Checks if a path exists. If it doesn't, create
    the path recursively
    """
    if not os.path.isdir(path):
        os.makedirs(path)