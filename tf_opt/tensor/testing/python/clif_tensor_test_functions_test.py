# coding=utf-8
# Copyright 2024 The tf.opt Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from google3.testing.pybase import googletest
from google3.third_party.tf_opt.tensor.testing.python import clif_tensor_test_functions


class SquareInplaceTest(googletest.TestCase):

  def testReduceSum(self):
    x = np.array([[1.0, 2.0], [5.0, 6.0]])
    result = clif_tensor_test_functions.TfOptTestReduceSum(x)
    self.assertAlmostEqual(14.0, result)

  def testTwoByThree(self):
    actual = clif_tensor_test_functions.TwoByThree(12.0)
    expected = np.array([[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]])
    np.testing.assert_allclose(expected, actual)


if __name__ == '__main__':
  googletest.main()
