from src.metrics.intrinsic_dimension import IntrinsicDimension
import unittest
import numpy as np
from unittest.mock import Mock

class TestIntrinsicDim(unittest.TestCase):
    def setUp(self):
        self.my_class = IntrinsicDimension(path="assets/fake_rep", parallel=False)
        self.mock_logger = Mock()

    def test_paralell_compute(self):
        mat_dist = np.random.rand(10, 200, 101).astype(np.float32)  # Fake input
        mat_dist *= 1000
        mat_coord = np.random.randint(0, 200, (10, 200, 101)).astype(np.int32)
        actual_output = self.my_class.parallel_compute(mat_dist, mat_coord)
        print(f"type = {type(actual_output)},\n{actual_output=}")
        # self.mock_logger.info.assert_called_with("Processing data in compute_fold")
        self.assertTrue(isinstance(actual_output, np.ndarray))
        

if __name__ == '__main__':
    unittest.main()
