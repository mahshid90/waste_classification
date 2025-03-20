import tensorflow as tf
from tensorflow import keras
import os

def load_model() :
    path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    full_path = os.path.join(path, 'models', 'custom_resnet50_model_v3.keras')
    model = tf.keras.models.load_model(full_path)
    return model
