# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import functools
import itertools
import operator
from functools import partial
from unittest import SkipTest

import jax
import jax.ops
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jax import numpy as jnp
from jax._src import dtypes
from jax._src import test_util as jtu
from jax.config import config

from ..utils.timer_wrapper import jax_op_timer, partial_timed

config.parse_flags_with_absl()
FLAGS = config.FLAGS

nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
one_dim_array_shapes = [(1,), (6,), (12,)]
empty_array_shapes = [
    (0,),
    (0, 4),
    (3, 0),
]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE, jtu.PYTHON_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes + empty_array_shapes
nonzerodim_shapes = nonempty_nonscalar_array_shapes + empty_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes = scalar_shapes + array_shapes

float_dtypes = jtu.dtypes.all_floating
complex_dtypes = jtu.dtypes.complex
int_dtypes = jtu.dtypes.all_integer
unsigned_dtypes = jtu.dtypes.all_unsigned
bool_dtypes = jtu.dtypes.boolean
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes + unsigned_dtypes
all_dtypes = number_dtypes + bool_dtypes


python_scalar_dtypes = [jnp.bool_, jnp.int_, jnp.float_, jnp.complex_]

# uint64 is problematic because with any uint type it promotes to float:
int_dtypes_no_uint64 = [d for d in int_dtypes + unsigned_dtypes if d != np.uint64]


def _valid_dtypes_for_shape(shape, dtypes):
    # Not all (shape, dtype) pairs are valid. In particular, Python scalars only
    # have one type in each category (float, bool, etc.)
    if shape is jtu.PYTHON_SCALAR_SHAPE:
        return [t for t in dtypes if t in python_scalar_dtypes]
    return dtypes


OpRecord = collections.namedtuple(
    "OpRecord",
    [
        "name",
        "nargs",
        "dtypes",
        "shapes",
        "rng_factory",
        "diff_modes",
        "test_name",
        "check_dtypes",
        "tolerance",
        "inexact",
        "kwargs",
    ],
)


def op_record(
    name,
    nargs,
    dtypes,
    shapes,
    rng_factory,
    diff_modes,
    test_name=None,
    check_dtypes=True,
    tolerance=None,
    inexact=False,
    kwargs=None,
):
    test_name = test_name or name
    return OpRecord(
        name,
        nargs,
        dtypes,
        shapes,
        rng_factory,
        diff_modes,
        test_name,
        check_dtypes,
        tolerance,
        inexact,
        kwargs,
    )


JAX_BITWISE_OP_RECORDS = [
    op_record(
        "bitwise_and",
        2,
        int_dtypes + unsigned_dtypes,
        all_shapes,
        jtu.rand_fullrange,
        [],
    ),
]


class _OverrideEverything:
    pass


class _OverrideNothing:
    pass


def _dtypes_are_compatible_for_bitwise_ops(args):
    if len(args) <= 1:
        return True
    is_signed = lambda dtype: jnp.issubdtype(dtype, np.signedinteger)
    width = lambda dtype: jnp.iinfo(dtype).bits
    x, y = args
    if width(x) > width(y):
        x, y = y, x
    # The following condition seems a little ad hoc, but seems to capture what
    # numpy actually implements.
    return (
        is_signed(x) == is_signed(y)
        or (width(x) == 32 and width(y) == 32)
        or (width(x) == 32 and width(y) == 64 and is_signed(y))
    )


def _shapes_are_broadcast_compatible(shapes):
    try:
        lax.broadcast_shapes(*(() if s in scalar_shapes else s for s in shapes))
    except ValueError:
        return False
    else:
        return True


def _shapes_are_equal_length(shapes):
    return all(len(shape) == len(shapes[0]) for shape in shapes[1:])


class JaxNumpyOperatorTests(jtu.JaxTestCase):
    """Tests for LAX-backed Numpy operators."""

    def _GetArgsMaker(self, rng, shapes, dtypes, np_arrays=True):
        def f():
            out = [
                rng(shape, dtype or jnp.float_) for shape, dtype in zip(shapes, dtypes)
            ]
            if np_arrays:
                return out
            return [
                jnp.asarray(a) if isinstance(a, (np.ndarray, np.generic)) else a
                for a in out
            ]

        return f

    @parameterized.parameters(
        itertools.chain.from_iterable(
            jtu.sample_product_testcases(
                [dict(name=rec.name, rng_factory=rec.rng_factory)],
                shapes=filter(
                    _shapes_are_broadcast_compatible,
                    itertools.combinations_with_replacement(rec.shapes, rec.nargs),
                ),
                dtypes=filter(
                    _dtypes_are_compatible_for_bitwise_ops,
                    itertools.combinations_with_replacement(rec.dtypes, rec.nargs),
                ),
            )
            for rec in JAX_BITWISE_OP_RECORDS
        )
    )
    @jax.numpy_rank_promotion(
        "allow"
    )  # This test explicitly exercises implicit rank promotion.
    def testBitwiseOp(self, name, rng_factory, shapes, dtypes):
        np_op = getattr(np, name)
        # timer = jax_op_timer()
        jnp_op = partial_timed(getattr(jnp, name))
        rng = rng_factory(self.rng())
        args_maker = self._GetArgsMaker(rng, shapes, dtypes)
        with jtu.strict_promotion_if_dtypes_match(dtypes):
            self._CheckAgainstNumpy(jtu.promote_like_jnp(np_op), jnp_op, args_maker)
            self._CompileAndCheck(jnp_op, args_maker)
