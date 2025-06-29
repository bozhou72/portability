import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer


class VariableOpTest(test.TestCase):
    
  def _initFetch(self, x, tftype, use_gpu=None):
    with self.test_session(use_gpu=use_gpu):
      p = state_ops.variable_op(x.shape, tftype)
      op = state_ops.assign(p, x)
      op.op.run()
      return self.evaluate(p)


  @test_util.run_deprecated_v1
  def testIsVariableInitialized(self):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        v0 = state_ops.variable_op([1, 2], dtypes.float32)
        timer = tensorflow_op_timer()
        with timer:
          test = variables.is_variable_initialized(v0)
          timer.gen.send(test)
        self.assertEqual(False, variables.is_variable_initialized(v0).eval())
        state_ops.assign(v0, [[2.0, 3.0]]).eval()
        timer = tensorflow_op_timer()
        with timer:
          test = variables.is_variable_initialized(v0)
          timer.gen.send(test)
        self.assertEqual(True, variables.is_variable_initialized(v0).eval())



if __name__ == "__main__":
  test.main()