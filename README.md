# Transfer Learning Using CNN and Random Forest

This project demonstrates how to leverage Transfer Learning by using a Convolutional Neural Network (CNN) as a feature extractor and Random Forest (RF) for classification.

## Files Overview

### `AnimalClassCNN`

This script utilizes a classical CNN model for image classification. The architecture consists of convolutional layers followed by dense layers:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

The resulting model weights are saved in animal_model.keras.

### `RFfromFeatureExtract`
This script uses the convolutional layers up to the flatten layer from the CNN as a custom feature extractor. The extracted features are then used with a Random Forest model for classification.

### `RFafterCNN`
This script utilizes the CNN model saved in animal_model.keras to extract features up to the flatten layer and then applies Random Forest for image classification.
