# Owner(s): ["module: nn"]

from torch.types import _TensorOrTensors
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_is_not_fp32, tf32_off, tf32_on
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
import torch.testing._internal.hypothesis_utils as hu
from torch.testing import make_tensor
from hypothesis import given
from torch.nn import MultiheadAttention
from torch.testing._internal.common_device_type import dtypes
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, TEST_CUDNN_VERSION
from torch.testing._internal.common_utils import freeze_rng_state, run_tests, TestCase, skipIfNoLapack, skipIfRocm, \
    skipIfRocmVersionLessThan, skipIfNotMiopenSuggestNHWC, TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMps,\
    suppress_warnings, TemporaryFileName, TEST_WITH_UBSAN, IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, set_default_dtype, IS_WINDOWS

from torch.testing._internal.common_dtype import integral_types, floating_types_and, get_all_math_dtypes, \
    floating_and_complex_types_and
from torch.nn.parallel._functions import Broadcast
from torch.nn.parameter import UninitializedParameter, UninitializedBuffer
from torch.nn import Parameter
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.utils.prune as prune
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch.nn.utils.rnn as rnn_utils
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.autograd.forward_ad as fwAD
from torch._six import inf, nan
import torch

import os
from ..utils.pytorch_device_decorators import onlyNativeDeviceTypes, onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer
# TODO: remove this global setting
# NN tests use double as the default dtype


AMPERE_OR_ROCM = TEST_WITH_ROCM or tf32_is_not_fp32()

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if TEST_SCIPY:
    from scipy import stats
    import scipy.signal
    import scipy.ndimage

if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.

class TestNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _padded_sequence(self, device, dtype):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(device, dtype)
        lengths = [len(i) for i in ordered]
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths

    @dtypes(torch.float32)
    def test_to(self, device, dtype):
        torch.set_default_dtype(torch.double)
        m = nn.Linear(3, 5)
        with pytorch_op_timer():
            m_cpu = m.to('cpu')
        self.assertIs(m, m_cpu)
        with pytorch_op_timer():
            result = m.to('cpu', dtype=torch.float32)
        self.assertIs(m, result)
        with pytorch_op_timer():
            result = m.to(torch.float64)
        self.assertEqual(m.double(), result)
        self.assertRaises(RuntimeError, lambda: m.to('cpu', copy=True))

        if os.environ['DEVICE'] != "cpu":
            with pytorch_op_timer():
                m2 = m.to(device)
            with pytorch_op_timer():
                result = m2.to(device)
            self.assertIs(m2, result)

            with pytorch_op_timer():
                m2.to('cpu')
            self.assertEqual(m, result)
            with pytorch_op_timer():
                result = m.to(device)
            self.assertEqual(m2, result)

            with pytorch_op_timer():
                result = m2.to(dtype=torch.float32)
            self.assertIs(m2, result)

            with pytorch_op_timer():
                result = m2.to(dtype=torch.float64)
            self.assertEqual(m2.double(), result)
        torch.set_default_dtype(torch.float)


instantiate_device_type_tests(TestNN, globals())

if __name__ == '__main__':
    run_tests()
