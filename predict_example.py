import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

"""
## Define constants
"""
img_width = 200
img_height = 50
max_length = 6  # Adjust this based on your dataset

# Characters present in the dataset must be as same as the one used in training
characters = sorted(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
                    'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

"""
## Load the model
"""
model = load_model('captcha_ocr_model', compile=False)

# Extract the prediction model
prediction_model = tf.keras.models.Model(model.get_layer(
    name="image").input, model.get_layer(name="dense2").output)
prediction_model.summary()

"""
## Preprocessing functions
"""
# Define a function to preprocess the image
def preprocess_image(img_path):
    # Read image
    img = tf.io.read_file(img_path)
    # Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # Expand dims to add batch size
    img = tf.expand_dims(img, axis=0)
    return img

# Define a function to decode the prediction
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

"""
## Inference
"""

# Load and preprocess the image
img_path = 'test.png'
img = preprocess_image(img_path)

# Make the prediction
pred = prediction_model.predict(img)
pred_text = decode_prediction(pred)

# Print the prediction
print("Predicted text:", pred_text[0])

# Visualize the image and prediction
plt.imshow(img[0, :, :, 0].numpy().T, cmap='gray')
plt.title(f"Prediction: {pred_text[0]}")
plt.axis('off')
plt.show()
