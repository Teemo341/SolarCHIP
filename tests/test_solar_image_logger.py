import unittest
from unittest import mock

from solarchip.utils.callback import SolarImageLogger


class SolarImageLoggerEpochFrequencyTest(unittest.TestCase):
    def setUp(self):
        self.logger = SolarImageLogger(
            every_n_train_epochs=3,
            max_images=2,
        )
        self.module = mock.Mock()
        self.module.global_step = 1
        self.trainer = mock.Mock(sanity_checking=False)

    def test_logs_train_and_validation_once_on_scheduled_epoch(self):
        self.module.current_epoch = 2

        with mock.patch.object(self.logger, "log_img") as log_img:
            for batch_idx in range(3):
                self.logger.on_train_batch_end(
                    self.trainer, self.module, None, "train-batch", batch_idx
                )
                self.logger.on_validation_batch_end(
                    self.trainer, self.module, None, "val-batch", batch_idx
                )

        self.assertEqual(log_img.call_count, 2)
        self.assertEqual(log_img.call_args_list[0].kwargs["split"], "train")
        self.assertEqual(log_img.call_args_list[1].kwargs["split"], "val")
        self.assertTrue(log_img.call_args_list[0].kwargs["force"])
        self.assertTrue(log_img.call_args_list[1].kwargs["force"])

    def test_skips_unscheduled_epoch_and_sanity_validation(self):
        with mock.patch.object(self.logger, "log_img") as log_img:
            self.module.current_epoch = 1
            self.logger.on_train_batch_end(
                self.trainer, self.module, None, "train-batch", 0
            )
            self.logger.on_validation_batch_end(
                self.trainer, self.module, None, "val-batch", 0
            )

            self.module.current_epoch = 2
            self.trainer.sanity_checking = True
            self.logger.on_validation_batch_end(
                self.trainer, self.module, None, "val-batch", 0
            )

        log_img.assert_not_called()

    def test_rejects_non_positive_epoch_frequency(self):
        with self.assertRaisesRegex(ValueError, "every_n_train_epochs"):
            SolarImageLogger(every_n_train_epochs=0)


if __name__ == "__main__":
    unittest.main()
