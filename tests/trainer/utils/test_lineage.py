import unittest
from unittest.mock import patch, MagicMock, call

from alpha_seed.trainer.utils.lineage import (safely_do, report_job_config, _detect_sdk_version, _SDKVersion, logger)


class TestDetectSDKVersion(unittest.TestCase):

    @patch('alpha_seed.trainer.utils.lineage.find_spec')
    def test_sdk_not_installed(self, mock_find_spec):
        mock_find_spec.return_value = None

        result = _detect_sdk_version()

        mock_find_spec.assert_called_once_with("bytedmerlin")
        self.assertIsNone(result)

    @patch('alpha_seed.trainer.utils.lineage.find_spec')
    @patch('alpha_seed.trainer.utils.lineage.version')
    def test_old_rpc_client_version(self, mock_version, mock_find_spec):
        # Given: Version less than 0.0.5.25
        mock_find_spec.return_value = MagicMock()
        mock_version.return_value = "0.0.5.20"

        # When:
        result = _detect_sdk_version()

        # Then:
        mock_find_spec.assert_called_once_with("bytedmerlin")
        mock_version.assert_called_once_with("bytedmerlin")
        self.assertEqual(result, _SDKVersion.RPC_CLIENT)

    @patch('alpha_seed.trainer.utils.lineage.find_spec')
    @patch('alpha_seed.trainer.utils.lineage.version')
    def test_http_client_without_builders(self, mock_version, mock_find_spec):
        # Given: bytedmerlin exists, Version greater than 0.0.5.25
        def mock_find_spec_side_effect(module_name):
            if module_name == "bytedmerlin":
                return MagicMock()
            else:
                return None

        mock_find_spec.side_effect = mock_find_spec_side_effect
        mock_version.return_value = "0.0.5.30"

        # When:
        result = _detect_sdk_version()

        # Then:
        self.assertEqual(mock_find_spec.call_count, 3)
        mock_version.assert_called_once_with("bytedmerlin")
        self.assertEqual(result, _SDKVersion.HTTP_CLIENT)

    @patch('alpha_seed.trainer.utils.lineage.find_spec')
    @patch('alpha_seed.trainer.utils.lineage.version')
    def test_http_client_with_lineage_builder(self, mock_version, mock_find_spec):
        # Given: bytedmerlin exists, Version greater than 0.0.5.25
        def mock_find_spec_side_effect(module_name):
            if module_name in ["bytedmerlin", "bytedmerlin.ai_assets.report_builder"]:
                return MagicMock()
            else:
                return None

        mock_find_spec.side_effect = mock_find_spec_side_effect
        mock_version.return_value = "0.0.6.0"

        result = _detect_sdk_version()

        self.assertEqual(mock_find_spec.call_count, 3)
        mock_version.assert_called_once_with("bytedmerlin")
        self.assertEqual(result, _SDKVersion.HTTP_CLIENT_WITH_LINEAGE_BUILDER)

    @patch('alpha_seed.trainer.utils.lineage.find_spec')
    @patch('alpha_seed.trainer.utils.lineage.version')
    def test_http_client_with_lineage_and_profiling(self, mock_version, mock_find_spec):

        def mock_find_spec_side_effect(module_name):
            return MagicMock()

        mock_find_spec.side_effect = mock_find_spec_side_effect
        mock_version.return_value = "0.0.6.0"

        result = _detect_sdk_version()

        self.assertEqual(mock_find_spec.call_count, 3)
        mock_version.assert_called_once_with("bytedmerlin")
        self.assertEqual(result, _SDKVersion.HTTP_CLIENT_WITH_LINEAGE_AND_PROFILING_BUILDER)


class TestSafelyDo(unittest.TestCase):

    @patch('alpha_seed.trainer.utils.lineage._detect_sdk_version')
    @patch('alpha_seed.trainer.utils.lineage.logger')
    def test_non_zero_rank(self, mock_logger, mock_detect_sdk_version):
        # Given: rank is not 0
        test_func = MagicMock()

        # When:
        wrapped = safely_do(test_func, rank=1)
        result = wrapped()

        # Then:
        test_func.assert_not_called()
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_not_called()
        self.assertIsNone(result)

    @patch('alpha_seed.trainer.utils.lineage._detect_sdk_version')
    @patch('alpha_seed.trainer.utils.lineage.logger')
    def test_sdk_not_installed(self, mock_logger, mock_detect_sdk_version):
        # Given: SDK is not installed
        mock_detect_sdk_version.return_value = None
        test_func = MagicMock()

        # When:
        wrapped = safely_do(test_func, rank=0)
        result = wrapped()

        # Then: Assert warning was logged and function was not called
        test_func.assert_not_called()
        mock_logger.warning.assert_called_once_with(
            "bytedmerlin is not installed, aborted lineage report; please install the latest version or use the image `data.aml.verl` with version v190 or higher",
            stacklevel=2)
        self.assertIsNone(result)

    @patch('alpha_seed.trainer.utils.lineage._detect_sdk_version')
    @patch('alpha_seed.trainer.utils.lineage.logger')
    def test_old_sdk_version(self, mock_logger, mock_detect_sdk_version):
        # Given: SDK is installed with old RPC client version
        mock_detect_sdk_version.return_value = _SDKVersion.RPC_CLIENT
        test_func = MagicMock()
        test_func.return_value = "test_result"

        # When:
        wrapped = safely_do(test_func, rank=0)
        result = wrapped()

        # Then: Assert warning was logged but function was still executed
        test_func.assert_called_once()
        mock_logger.warning.assert_called_once_with(
            "old rpc bytedmerlin is detected; please upgrade your bytedmerlin to the latest version or use the image `data.aml.verl` with version v190 or higher",
            stacklevel=2)
        mock_logger.info.assert_called_once_with("lineage reported: test_result", stacklevel=2)
        self.assertEqual(result, "test_result")

    @patch('alpha_seed.trainer.utils.lineage._detect_sdk_version')
    @patch('alpha_seed.trainer.utils.lineage.logger')
    def test_newer_sdk_version(self, mock_logger, mock_detect_sdk_version):
        # Given: SDK is installed with newer SDK version
        mock_detect_sdk_version.return_value = _SDKVersion.HTTP_CLIENT_WITH_LINEAGE_BUILDER
        test_func = MagicMock()
        test_func.return_value = "success_result"

        # When:
        wrapped = safely_do(test_func, rank=0)
        result = wrapped()

        # Then: Assert function was called and success was logged
        test_func.assert_called_once()
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_called_once_with("lineage reported: success_result", stacklevel=2)
        self.assertEqual(result, "success_result")

    @patch('alpha_seed.trainer.utils.lineage._detect_sdk_version')
    @patch('alpha_seed.trainer.utils.lineage.logger')
    def test_exception_handling(self, mock_logger, mock_detect_sdk_version):
        # Given: SDK is installed with newer SDK version, but function raises an exception
        mock_detect_sdk_version.return_value = _SDKVersion.HTTP_CLIENT
        test_func = MagicMock()
        test_func.side_effect = ValueError("Test exception")

        # When:
        wrapped = safely_do(test_func, rank=0)
        result = wrapped()

        # Then: Assert exception was caught and logged
        test_func.assert_called_once()
        mock_logger.warning.assert_called_once()
        self.assertIn("error is raised when trying to report lineage safely", mock_logger.warning.call_args[0][0])
        self.assertIsNone(result)

    @patch('alpha_seed.trainer.utils.lineage.OmegaConf')
    @patch('bytedmerlin.ai_assets.report_builder.MerlinJobReportBuilder')
    @patch('alpha_seed.trainer.utils.lineage._detect_sdk_version')
    def test_integration_with_report_job_config(self, mock_detect_sdk_version, mock_builder_class, mock_omega_conf):
        mock_detect_sdk_version.return_value = _SDKVersion.HTTP_CLIENT_WITH_LINEAGE_BUILDER
        mock_config = {"test": "config"}
        mock_container = {"converted": "config"}
        mock_omega_conf.to_container.return_value = mock_container

        mock_builder = MagicMock()
        mock_builder_class.return_value = mock_builder
        mock_builder.report.return_value = "report_success"

        result = safely_do(lambda: report_job_config(mock_config), rank=0)()

        mock_omega_conf.to_container.assert_called_once_with(mock_config)
        mock_builder_class.assert_called_once()
        mock_builder.report.assert_called_once_with("alpha-seed", is_async=True, job_config=mock_container)
        self.assertEqual(result, "report_success")


if __name__ == '__main__':
    unittest.main()
