#Triplet Network 
#Importing Dependencies
import os, random, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import mixed_precision 
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Lambda 
from tensorflow.keras.models import Model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Configuration
mixed_precision.set_global_policy('mixed_float16')
IMG_SIZE = (224, 224)
INPUT_SHAPE = IMG_SIZE + (3,)
MARGIN = 0.3
BATCH_SIZE = 16
EPOCHS = 25
TRAIN_DIR = "Comys_Hackathon5/Task_B/train"
VAL_DIR = "Comys_Hackathon5/Task_B/val"
SAVE_PATH = "TripletNetwork.h5"

#Triplet Loss Function
def triplet_loss(margin=MARGIN):
    def _loss(y_true, y_pred):
        a, p, n = y_pred[:, :256], y_pred[:, 256:512], y_pred[:, 512:]
        pos_dist = tf.reduce_sum(tf.square(a - p), axis=1)
        neg_dist = tf.reduce_sum(tf.square(a - n), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return _loss

#Embedding Model
def create_embedding_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256)(x)
    x = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs=base.input, outputs=x), base

# Building Triplet Model
def build_triplet_model():
    embedding_model, base = create_embedding_model()
    in_a, in_p, in_n = Input(INPUT_SHAPE), Input(INPUT_SHAPE), Input(INPUT_SHAPE)
    emb_a = embedding_model(in_a)
    emb_p = embedding_model(in_p)
    emb_n = embedding_model(in_n)
    merged = tf.keras.layers.concatenate([emb_a, emb_p, emb_n])
    model = Model([in_a, in_p, in_n], merged)
    model.compile(optimizer='adam', loss=triplet_loss())
    return model, embedding_model, base

#Preprocessing and Augmentation

def preprocess_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return tf.keras.applications.resnet50.preprocess_input(img)

#Triplet Generator 
def generate_triplets(data_dir, batch_size=BATCH_SIZE):
    ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    get_images = lambda p: [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('jpg','jpeg','png'))]
    while True:
        a, p, n = [], [], []
        for _ in range(batch_size):
            pos_id = random.choice(ids)
            neg_id = random.choice([i for i in ids if i != pos_id])
            pos_dir = os.path.join(data_dir, pos_id)
            anchor_imgs = get_images(pos_dir)
            dist_imgs = get_images(os.path.join(pos_dir, "distortion"))
            neg_imgs = get_images(os.path.join(data_dir, neg_id))
            if not (anchor_imgs and dist_imgs and neg_imgs): continue
            a.append(preprocess_image(random.choice(anchor_imgs)))
            p.append(preprocess_image(random.choice(dist_imgs)))
            n.append(preprocess_image(random.choice(neg_imgs)))
        yield (np.array(a, np.float32), np.array(p, np.float32), np.array(n, np.float32)), np.zeros((batch_size, 768), np.float32)

#Tensorflow Dataset Wrapper
def wrap_generator(data_dir):
    sig = ((tf.TensorSpec((None, *INPUT_SHAPE), tf.float32),)*3, tf.TensorSpec((None, 768), tf.float32))
    return tf.data.Dataset.from_generator(lambda: generate_triplets(data_dir), output_signature=sig).repeat().prefetch(tf.data.AUTOTUNE)

#Training and  Fine-tuning
def train():
    model, embed, base = build_triplet_model()
    train_ds = wrap_generator(TRAIN_DIR)
    val_ds = wrap_generator(VAL_DIR)
    history = model.fit(train_ds, steps_per_epoch=500, epochs=EPOCHS, validation_data=val_ds, validation_steps=100)

    # Fine-tune top layers
    base.trainable = True
    for l in base.layers[:-10]: l.trainable = False
    model.compile(optimizer='adam', loss=triplet_loss())
    model.fit(train_ds, steps_per_epoch=250, epochs=5)

    model.save(SAVE_PATH)
    plot_metrics(history.history)
    return embed

#Plot Training Curves
def plot_metrics(hist):
    plt.figure(figsize=(10,4))
    plt.plot(hist['loss'], label='Train Loss')
    plt.plot(hist['val_loss'], label='Val Loss')
    plt.legend(); plt.grid(); plt.title("Triplet Loss"); plt.xlabel("Epochs")
    plt.savefig("triplet_training_loss.png"); plt.show()

#Embedding Extractor
def extract_embedding(model, img_path):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)
    return model.predict(img, verbose=0)[0]

#Threshold Tuning
def compute_threshold(embed_model):
    distances, labels = [], []
    ids = [d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]
    get_images = lambda p: [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('jpg','jpeg','png'))]

    for id in tqdm(ids):
        id_path = os.path.join(VAL_DIR, id)
        ref_img = random.choice(get_images(id_path))
        ref_emb = extract_embedding(embed_model, ref_img)
        for dist_img in get_images(os.path.join(id_path, "distortion")):
            test_emb = extract_embedding(embed_model, dist_img)
            distances.append(np.linalg.norm(ref_emb - test_emb)); labels.append(1)
        neg_id = random.choice([nid for nid in ids if nid != id])
        neg_img = random.choice(get_images(os.path.join(VAL_DIR, neg_id)))
        neg_emb = extract_embedding(embed_model, neg_img)
        distances.append(np.linalg.norm(ref_emb - neg_emb)); labels.append(0)

    best_acc, best_thresh = 0, 0
    for t in np.linspace(0.2, 1.5, 300):
        preds = [1 if d < t else 0 for d in distances]
        acc = accuracy_score(labels, preds)
        if acc > best_acc: best_acc, best_thresh = acc, t

    preds = [1 if d < best_thresh else 0 for d in distances]
    print(f"\n\u2705 Best Threshold: {best_thresh:.4f} (Accuracy: {best_acc:.4f})")
    print(f"Precision: {precision_score(labels, preds):.4f}, Recall: {recall_score(labels, preds):.4f}, F1: {f1_score(labels, preds):.4f}")

if __name__ == '__main__':
    emb_model = train()
    compute_threshold(emb_model)
