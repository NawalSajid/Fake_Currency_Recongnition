import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Paths to real and fake currency folders
real_path = '/UNIVERSITY/PythonDIP/dataset/real'   
fake_path = '/UNIVERSITY/PythonDIP/dataset/fake'

# Load images safely from folders
def load_images(folder, label, img_size=(100, 100), walk_subfolders=False):
    images, labels = [], []
    valid_exts = ('.jpg', '.jpeg', '.png')

    if walk_subfolders:
        for root, dirs, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith(valid_exts):
                    fpath = os.path.join(root, fname)
                    try:
                        img = load_img(fpath, target_size=img_size)
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(label)
                    except:
                        continue
    else:
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if not os.path.isfile(fpath) or not fname.lower().endswith(valid_exts):
                continue
            try:
                img = load_img(fpath, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except:
                continue

    return images, labels

# Load real and fake notes
real_imgs, real_labels = load_images(real_path, 0, walk_subfolders=True)
fake_imgs, fake_labels = load_images(fake_path, 1, walk_subfolders=False)

# Combine and prepare data
X = np.array(real_imgs + fake_imgs)
y = np.array(real_labels + fake_labels)
y = to_categorical(y, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate on test set
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# Classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Save model
model.save('currency_model.keras')
print("Model saved successfully!")
