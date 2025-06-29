# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tensorflow.ops.reshape_op."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test
from ..utils.timer_wrapper import tensorflow_op_timer


@test_util.with_eager_op_as_function
class ReshapeTest(test.TestCase):

  def _testReshape(self, x, y):
    with self.cached_session():
      timer = tensorflow_op_timer()
      with timer:
        np_ans = x.reshape(y)
        timer.gen.send(np_ans)
        timer = tensorflow_op_timer()
      with timer:
        tf_ans = array_ops.reshape(x, y)
        timer.gen.send(tf_ans)
      out = self.evaluate(tf_ans)
      self.assertEqual(tf_ans.get_shape(), out.shape)
      self.assertShapeEqual(np_ans, tf_ans)

      # Repeat with an int64 shape tensor.
      y64 = constant_op.constant(y, dtype=dtypes.int64)
      timer = tensorflow_op_timer()
      with timer:
        tf_ans = array_ops.reshape(x, y64)
        timer.gen.send(tf_ans)
      out = self.evaluate(tf_ans)
      self.assertEqual(tf_ans.get_shape(), out.shape)
      self.assertShapeEqual(np_ans, tf_ans)

  def _testZeroDimReshape(self, x, shape, expected):
    with self.cached_session():
      timer = tensorflow_op_timer()
      with timer:
        y = array_ops.reshape(x, shape)
        timer.gen.send(y)
      out = self.evaluate(y)
      self.assertEqual(expected, out.shape)

      # Repeat with an int64 shape tensor.
      shape64 = constant_op.constant(shape, dtype=dtypes.int64)
      timer = tensorflow_op_timer()
      with timer:
        y = array_ops.reshape(x, shape64)
        timer.gen.send(y)
      out = self.evaluate(y)
      self.assertEqual(expected, out.shape)

  def _testBothReshape(self, x, y):
    self._testReshape(x, y)
    self._testReshape(x, y)

  def testBoolBasic(self):
    timer = tensorflow_op_timer()
    with timer:
      x = np.arange(1., 7.).reshape([1, 6]) > 3
      timer.gen.send(x)
    self._testBothReshape(x, [2, 3])

  def testFloatBasic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float32)
    self._testBothReshape(x, [2, 3])

  def testDoubleBasic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float64)
    self._testBothReshape(x, [2, 3])

  def testInt32Basic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.int32)
    self._testBothReshape(x, [2, 3])

  def testComplex64Basic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.complex64)
    self._testBothReshape(x, [2, 3])

  # def testComplex128Basic(self):
  #   x = np.arange(1., 7.).reshape([1, 6]).astype(np.complex128)
  #   self._testBothReshape(x, [2, 3])

  def testFloatReshapeThreeDimensions(self):
    x = np.arange(1., 28.).reshape([1, 27]).astype(np.float32)
    self._testBothReshape(x, [3, 3, 3])

  def testFloatUnspecifiedDimOnly(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [-1])

  def testFloatUnspecifiedDimBegin(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [-1, 2])

  def testFloatUnspecifiedDimEnd(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [3, -1])

  def testZeroDimBasic(self):
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [0, 2, 3])

  def testZeroDimReshapeR1(self):
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [-1])

  def testZeroDimReshapeR3(self):
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [-1, 2, 3])

  # TODO(vrv): Add tests for failure conditions once python test_util
  # reports errors.

  def testFloatReshapeGradThreeDimensions(self):
    x = np.arange(1., 25.).reshape([2, 3, 4]).astype(np.float32)
    input_tensor = constant_op.constant(x)

    def reshape(x):
      timer = tensorflow_op_timer()
      with timer:
        test = array_ops.reshape(x, [1, 8, 3])
        timer.gen.send(test)
      return array_ops.reshape(x, [1, 8, 3])

    with self.cached_session():
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(reshape, [input_tensor]))
      self.assertLess(err, 1e-3)

  def testFloatEmpty(self):
    x = np.empty((0, 0, 0, 0), dtype=np.float32)
    self._testBothReshape(x, [1, 2, 3, 0])
    self._testBothReshape(x, [1, 0, 0, 4])
    self._testBothReshape(x, [0, 0, 0, 0])
    self._testBothReshape(x, [1, 2, 0])
    self._testBothReshape(x, [0, 0, 0])
    self._testBothReshape(x, [1, -1, 5])

  def testZeroDimWithUnspecifiedDim(self):
    # for use_gpu in (True, False):
    self._testZeroDimReshape(x=np.zeros([0, 6]).astype(np.float32),
                              shape=[0, -1, 3],
                              expected=(0, 2, 3))

  @test_util.run_deprecated_v1
  def testErrors(self):
    y = constant_op.constant(0.0, shape=[23, 29, 31])
    with self.assertRaisesRegex(ValueError, "must be evenly divisible by 17"):
      array_ops.reshape(y, [17, -1])

    z = constant_op.constant(0.0, shape=[32, 128])
    with self.assertRaisesRegex(ValueError,
                                "Cannot reshape a tensor with 4096 elements"):
      array_ops.reshape(z, [4095])

  def testPartialShapes(self):

    # Testing unknown shapes in graph building.
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)

      # Unknown input shape, partial new shape.
      timer = tensorflow_op_timer()
      with timer:
        y = array_ops.reshape(x, [1, 1, -1, 1])
        timer.gen.send(y)
      self.assertEqual([1, 1, None, 1], y.get_shape().as_list())

      # Unknown input shape, unknown new shape.
      timer = tensorflow_op_timer()
      with timer:
        y = array_ops.reshape(x, array_ops.placeholder(dtypes.int32))
        timer.gen.send(y)
      self.assertEqual(None, y.get_shape().ndims)

      # Unknown input shape, known rank for new shape.
      timer = tensorflow_op_timer()
      with timer:
        y = array_ops.reshape(x, array_ops.placeholder(dtypes.int32, shape=(3,)))
        timer.gen.send(y)
      self.assertEqual([None, None, None], y.get_shape().as_list())

      # Unknown input shape, partial new shape using `tf.stack()`.
      timer = tensorflow_op_timer()
      with timer:
        y = array_ops.reshape(x, [array_ops.placeholder(dtypes.int32), 37])
        timer.gen.send(y)
      self.assertEqual([None, 37], y.get_shape().as_list())

      # Unknown input shape, partial new shape using `tf.concat()`.
      timer = tensorflow_op_timer()
      with timer:
        y = array_ops.reshape(
          x,
          array_ops.concat(
              [array_ops.placeholder(
                  dtypes.int32, shape=(2,)), [37, 42]], 0))
        timer.gen.send(y)
      self.assertEqual([None, None, 37, 42], y.get_shape().as_list())

      # Unknown input shape, partial new shape using `tf.shape()`.
      timer = tensorflow_op_timer()
      with timer:
        y = array_ops.reshape(
          x,
          array_ops.shape(
              array_ops.placeholder(
                  dtypes.float32, shape=[None, 37, None])))
        timer.gen.send(y)
      self.assertEqual([None, 37, None], y.get_shape().as_list())

  def testTensorShape(self):
    x = array_ops.zeros([1, 100])
    timer = tensorflow_op_timer()
    with timer:
      y = array_ops.reshape(
        x, [tensor_shape.Dimension(100),
            tensor_shape.Dimension(1)])
      timer.gen.send(y)
    self.assertEqual([100, 1], y.get_shape().as_list())
    timer = tensorflow_op_timer()
    with timer:
      y = array_ops.reshape(x, tensor_shape.TensorShape([100, 1]))
      timer.gen.send(y)
    self.assertEqual([100, 1], y.get_shape().as_list())

  def testInt64Shape(self):
    # with ops.device("/device:CPU:0"):
    x = array_ops.zeros([50000, 50000], dtype=dtypes.bool)
    # Provide dimension larger than int32
    timer = tensorflow_op_timer()
    with timer:
      y = array_ops.reshape(x, [50000**2])
      timer.gen.send(y)
    self.assertEqual([50000**2], y.get_shape().as_list())
    # Even if first dimension is within int32, ensure we correctly go to int64
    timer = tensorflow_op_timer()
    with timer:
      y = array_ops.reshape(x, [1, 50000**2])
      timer.gen.send(y)
    self.assertEqual([1, 50000**2], y.get_shape().as_list())

  # @test_util.run_v2_only
  # def testTooLargeShape(self):
  #   with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
  #                               "too many elements"):
  #     with tensorflow_op_timer():
  #       x = array_ops.reshape([1], np.array([21943, 45817, 30516, 61760, 38987]))
  #     self.evaluate(x)


if __name__ == "__main__":
  test.main()