from utils import get_device, get_batch_size

class Config:
    DEVICE = get_device()
    BATCH_SIZE = get_batch_size(DEVICE)
    CHECKPOINTS_PATH = './models/checkpoints/fracturenet_checkpoints'
    TRAIN_IMAGES_PATH = './dataset/train_images'
    TEST_IMAGES_PATH = './dataset/test_images'
    MODEL_NAMES = [f'f{i}' for i in range(5)]
    FRAC_COLS = [f'C{i}_frac' for i in range(1, 8)]
    VERT_COLS = [f'C{i}_vert' for i in range(1, 8)]