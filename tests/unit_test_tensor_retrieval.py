from src.utils.tensor_storage import retrieve_from_storage
import unittest
import numpy as np
from unittest.mock import Mock
import pandas as pd

class TestTensorRetrieval(unittest.TestCase):
    def setUp(self):
        self.my_function = retrieve_from_storage
        

    def test_retrieve_from_storage(self):
        hidden_states, logits, df = self.my_function("tests/assets/fake_rep")
        self.assertTrue(isinstance(hidden_states, np.ndarray))
        self.assertTupleEqual(hidden_states.shape, (3,10,5))
        self.assertTrue(isinstance(logits, np.ndarray))
        self.assertTrue(isinstance(df, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()
