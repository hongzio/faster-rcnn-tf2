from frcnn.faster_rcnn import FasterRCNN
from frcnn.config import Config

if __name__ == '__main__':
    cfg = Config()
    faster_rcnn = FasterRCNN(cfg)
    cfg['train']['rpn_batch_size'] = 1
    faster_rcnn.test('./frcnn/dataset/girl.png')