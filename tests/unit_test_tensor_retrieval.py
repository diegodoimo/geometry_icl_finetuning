from src.utils.tensor_storage import retrieve_from_storage
import unittest
from pathlib import Path
import numpy as np
from unittest.mock import Mock
import pandas as pd

class TestTensorRetrieval(unittest.TestCase):
    def setUp(self):
        self.my_function = retrieve_from_storage
        

    # def test_retrieve_from_storage_mat_distances(self):
    #     _PATH = Path("/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens"
    #             "/repo/results/evaluated_test/random_order/llama-3-8b/0shot")
    #     out = self.my_function(_PATH, -1,  False)
    #     tensors, labels, num_layers = out
    #     self.assertTrue(isinstance(tensors, tuple))
    #     self.assertTrue(isinstance(labels, dict))
    #     self.assertTrue(isinstance(num_layers, int))
    
    def test_retrieve_from_storage_full_tensors(self):
        _PATH = Path("/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens"
                "/repo/results/evaluated_test/random_order/llama-3-8b/0shot")
        out = self.my_function(_PATH, 200,  True)
        tensors, labels, num_layers = out
        self.assertTrue(isinstance(tensors, np.ndarray), msg=f"tensors is {type(tensors)}")
        self.assertTrue(isinstance(labels, dict))
        self.assertTrue(isinstance(num_layers, int))
    

if __name__ == '__main__':
    unittest.main()
