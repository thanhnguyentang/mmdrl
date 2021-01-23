
class IterationStatistics(object):
    def __init__(self):
        self.data_lists = {} 
    
    def append(self, data_pairs):
        for key, value in data_pairs.items():
            if key not in self.data_lists:
                self.data_lists[key] = [] 
            self.data_lists[key].append(value) 

