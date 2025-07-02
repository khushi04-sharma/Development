import tensorflow as tf

# Load the full triplet model
triplet_model = tf.keras.models.load_model(
    "TripletNetwork.h5",
    compile=False 
)

# Locate the embedding model
embedding_model = None
for layer in triplet_model.layers:
    if isinstance(layer, tf.keras.Model) and layer.output_shape[-1] == 256:
        embedding_model = layer
        break

if embedding_model:
    embedding_model.save("Embedding_Seq.h5")
    print("✅ Saved Embedding_Seq.h5 successfully.")
else:
    print("❌ Embedding model not found in triplet_model.")
