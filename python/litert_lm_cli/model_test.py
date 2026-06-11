# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

import litert_lm
from litert_lm_cli import model


class ModelTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('default_backend', None, None, 'cpu', None),
      ('cpu_backend', 'cpu', None, 'cpu', None),
      ('gpu_backend', 'gpu', None, 'gpu', None),
      ('npu_backend', 'npu', None, 'npu', None),
      ('cpu_with_threads', 'cpu', 4, 'cpu', 4),
      ('default_with_threads', None, 4, 'cpu', 4),
  )
  @mock.patch.object(model, '_backend_constraint')
  def test_parse_backend(
      self,
      backend,
      cpu_thread_count,
      expected_type_str,
      expected_thread_count,
      mock_backend_constraint,
  ):
    mock_backend_constraint.return_value = litert_lm.Backend.CPU()

    # Mock NPU to avoid RuntimeError on Linux
    with mock.patch.object(
        litert_lm.Backend, 'NPU', autospec=True
    ) as mock_npu_class:
      result = model.parse_backend(
          backend=backend, cpu_thread_count=cpu_thread_count
      )

      if expected_type_str == 'cpu':
        self.assertIsInstance(result, litert_lm.Backend.CPU)
        self.assertEqual(result.thread_count, expected_thread_count)
      elif expected_type_str == 'gpu':
        self.assertIsInstance(result, litert_lm.Backend.GPU)
      elif expected_type_str == 'npu':
        mock_npu_class.assert_called_once()
        self.assertEqual(result, mock_npu_class.return_value)

  @mock.patch.object(model, '_backend_constraint')
  def test_parse_backend_gpu_constraint(self, mock_backend_constraint):
    mock_backend_constraint.return_value = litert_lm.Backend.GPU()
    mock_model = mock.Mock(spec=model.Model)
    mock_model.model_path = 'dummy_path'

    result = model.parse_backend(backend='cpu', model_obj=mock_model)
    self.assertIsInstance(result, litert_lm.Backend.GPU)


if __name__ == '__main__':
  absltest.main()
