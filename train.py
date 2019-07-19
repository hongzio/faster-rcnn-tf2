from frcnn.faster_rcnn import FasterRCNN
from frcnn.config import Config

if __name__ == '__main__':
    cfg = Config()
    faster_rcnn = FasterRCNN(cfg)
    faster_rcnn.train()