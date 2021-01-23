import os
import pickle 

class Logger(object):
    """Class for logging data. 
    """
    def __init__(self, logging_dir, prefix, max_keep=4):
        self._logging_dir = logging_dir
        if not os.path.exists(self._logging_dir):
            os.makedirs(self._logging_dir)
        self.data = {}
        self._prefix = prefix
        self._max_keep = max_keep 
        self._iter = 0 
    def append(self, key, statistics):
        self.data[key] = statistics   

    def log_to_file(self, iteration):
        with open(os.path.join(self._logging_dir, '%s_%d'%(self._prefix, iteration)), 'wb') as fo:
            pickle.dump(self.data, fo, pickle.HIGHEST_PROTOCOL)  
        if iteration >= self._max_keep:
            iter_to_be_removed =  iteration - self._max_keep
            try:
                os.remove(os.path.join(self._logging_dir, '%s_%d'%(self._prefix, iter_to_be_removed))) 
            except:
                pass 
    def file_to_log(self, iteration):
        with open(os.path.join(self._logging_dir, '%s_%d'%(self._prefix, iteration)), 'rb') as fo:
            return pickle.load(fo) 