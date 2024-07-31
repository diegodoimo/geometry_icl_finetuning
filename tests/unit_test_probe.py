from src.metrics.probe import LinearProbe
from pathlib import Path
import unittest
import numpy as np
from unittest.mock import Mock

class TestProbe(unittest.TestCase):
    def setUp(self):
        self.my_class = LinearProbe(path="assets/fake_rep")
        self.mock_logger = Mock()

    # def test_paralell_compute(self):
    #     mat_dist = np.random.rand(10, 200, 101).astype(np.float32)  # Fake input
    #     mat_dist *= 1000
    #     mat_coord = np.random.randint(0, 200, (10, 200, 101)).astype(np.int32)
    #     label = np.random.randint(0, 10, (200)).astype(np.float32)  # Fake target
        
    #     actual_output = self.my_class.parallel_compute(mat_dist,
    #                                                    mat_coord,
    #                                                    label,
    #                                                    z=0.1)
    #     print(f"type = {type(actual_output)},\n{actual_output=}")
    #     # self.mock_logger.info.assert_called_with("Processing data in compute_fold")
    #     self.assertTrue(isinstance(actual_output, dict))
    
    def test_main(self):
        _PATH = Path("/orfeo/cephfs/scratch/area/ddoimo/open/geometric_lens"
                     "/repo/results/evaluated_test/random_order/llama-3-8b")

        probe = LinearProbe(path=_PATH / '2shot', parallel=True)
        out = probe.main(label="subjects", instance_per_sub=-1)
        self.assertTrue(isinstance(out, np.ndarray))

if __name__ == '__main__':
    unittest.main()
