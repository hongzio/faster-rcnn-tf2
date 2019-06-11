import unittest

from frcnn.config.config import Config


class TestConfig(unittest.TestCase):
    def test_epoch_config(self):
        cfg = Config('config/default.yml')
        self.assertEqual(cfg['train']['epoch'], 80)

    def test_write_config(self):
        cfg = Config('config/default.yml')
        cfg['train']['epoch'] = 70
        self.assertEqual(cfg['train']['epoch'], 70)


if __name__ == '__main__':
    unittest.main()
