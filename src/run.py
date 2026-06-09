from pipeline import tf_asl
import os

if __name__ == "__main__":
    DATA_DIR = os.path.join("..", "asl_dataset")
    IMAGE_SIZE = (128, 128)
    

    tf_asl(data_dir=DATA_DIR, image_size=IMAGE_SIZE)
