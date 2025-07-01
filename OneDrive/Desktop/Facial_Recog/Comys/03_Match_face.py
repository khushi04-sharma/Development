import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  
from tensorflow.keras.preprocessing.image import load_img, img_to_array 

#Configuration
EMBEDDING_MODEL_PATH = "Embedding_Seq.h5"
THRESHOLD = 0.945  
IMG_SIZE = (224, 224)

#Load Embedding Model
embedding_model = load_model(EMBEDDING_MODEL_PATH, compile=False)

#Preprocessing Function
def preprocess_image(img_path, target_size=IMG_SIZE):
    img = load_img(img_path, target_size=target_size)
    img = img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return np.expand_dims(img, axis=0)

#Get Embedding
def get_embedding(image_path):
    image = preprocess_image(image_path)
    embedding = embedding_model.predict(image)[0]
    return embedding / np.linalg.norm(embedding)  

#Match Function
def is_match(reference_path, test_path, threshold=THRESHOLD):
    ref_embedding = get_embedding(reference_path)
    test_embedding = get_embedding(test_path)
    distance = np.linalg.norm(ref_embedding - test_embedding)
    print(f"🔍 Distance = {distance:.4f}")
    return distance < threshold

#Example Usage
if __name__ == "__main__":
    reference_img = "050_frontal_foggy.jpg"
    test_img = "050_frontal_rainy.jpg"

    if is_match(reference_img, test_img):
        print("✅ MATCH: Same identity")
    else:
        print("❌ NO MATCH: Different identity")
