from src.metrics.clustering import LabelClustering
import unittest
import numpy as np
from unittest.mock import Mock

class TestClustering(unittest.TestCase):
    def setUp(self):
        self.my_class = LabelClustering(path="assets/fake_rep")
        self.mock_logger = Mock()

    def test_paralell_compute(self):
        mat_dist = np.random.rand(10, 100, 100).astype(np.float32)  # Fake input
        mat_coord = np.random.rand(10, 100, 100).astype(np.float32)
        label = np.random.randint(0, 10, (100)).astype(np.float32)  # Fake target
        
        actual_output = self.my_class.parallel_compute(mat_dist,
                                                       mat_coord,
                                                       label,
                                                       z=0.1)
        print(f"type = {type(actual_output)},\n{actual_output=}")
        # self.mock_logger.info.assert_called_with("Processing data in compute_fold")
        self.assertTrue(isinstance(actual_output, dict))
        

if __name__ == '__main__':
    unittest.main()
