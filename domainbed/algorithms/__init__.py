from .algorithms import *

def get_algorithm_class(algorithm_name):
    '''
    Return the algorithm class with the given name
    '''
    if algorithm_name not in globals():
        raise NotImplementedError(f'Algorithm not found : {algorithm_name}')
    return globals()[algorithm_name]