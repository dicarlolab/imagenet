__author__ = 'headradio'
#Some utilities to set up a mongdb instance with all the images
# from the collection of tar files available for download
import os
import tarfile
from joblib import Parallel, delayed

def get_tar_filenames(path=None):
    if path is None:
        path = os.getcwd()
    return [filename for filename in os.listdir() if filename.endswith('.tar')]


def put_tar_file_contents_on_gridfs(tar_file_name, fs):
    tar_file = tarfile.open(tar_file_name, mode='r|')
    filenames = []
    for member in tar_file:
        file = tar_file.extractfile()
        filename = member.name
        filenames.append(filename)
        fs.put(file, _id=filename)
    return filenames


def parallel_upload_tar_folder(fs, path=None, n_jobs=-1):
    if path is None:
        path = os.getcwd()
    tar_filenames = get_tar_filenames(path)
    ids = Parallel(n_jobs=n_jobs)(delayed(put_tar_file_contents_on_gridfs)(tar_filename, fs)
                                      for tar_filename in tar_filenames)
    return tar_filenames, ids

