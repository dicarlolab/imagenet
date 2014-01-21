__author__ = 'headradio'
#Some utilities to set up a mongdb instance with all the images
# from the collection of tar files available for download
import os
import tarfile
from joblib import Parallel, delayed


def get_tar_filenames(path=None):
    if path is None:
        path = os.getcwd()
    return [filename for filename in os.listdir(path) if filename.endswith('.tar')]


#for example:
#IMAGENET_DB_PORT = int(os.environ.get('IMAGENET_DB_PORT', 27017))
#db = pm.MongoClient('localhost', port=IMAGENET_DB_PORT).gridfs_example
#fs = gridfs.GridFS(db)


def put_tar_file_contents_on_gridfs(tar_file_name, fs, force=True):
    print 'uploading '+tar_file_name
    tar_file = tarfile.open(tar_file_name, mode='r|')
    filenames = []
    for member in tar_file:
        file_obj = tar_file.extractfile(member)
        filename = member.name
        filenames.append(filename)
        if force and fs.exists(filename):
            fs.delete(filename)
            print 'Overwriting '+filename
        fs.put(file_obj, _id=filename)
    return filenames


def parallel_upload_tar_folder(path=None, n_jobs=-1):
    """
    Utility to upload contents of many tarballs to the default gridfs
    :param path: Path to folder full of tars to upload
    :param n_jobs: Number of processes to use
    :return: returns tar filenames uplaoded and a list of lists of filenames (one per tar)
    """
    if path is None:
        path = os.getcwd()
    tar_filenames = get_tar_filenames(path)
    ids = Parallel(n_jobs=n_jobs)(delayed(put_tar_file_contents_on_gridfs)(tar_filename)
                                  for tar_filename in tar_filenames)
    return tar_filenames, ids


#took about 4 hours with all 24 processors
def parallel_filename_dict_from_tar_folder(path=None, n_jobs=-1):
    if path is None:
        path = os.getcwd()
    tar_filenames = get_tar_filenames(path)
    results = Parallel(n_jobs=n_jobs)(delayed(get_names)(tar_filename)
                                      for tar_filename in tar_filenames)
    return tar_filenames, results


def get_names(tarfilename):
    print 'scanning '+tarfilename
    return tarfile.open(tarfilename).getnames()
