# main.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

# --- Diagnostics: Python/TensorFlow environment ---
import sys
try:
    import tensorflow as tf
    print(f"[Diag] Python: {sys.version.split()[0]}  |  TF: {tf.__version__}")
    print("[Diag] Devices:", tf.config.list_physical_devices())
except Exception as e:
    print("[Diag] TensorFlow import failed:", repr(e))

# --- Settings ---
IMG_SIZE = (48, 48)
BATCH_SIZE = 64          # größer = stabilere Gradienten
EPOCHS = 25
MODEL_PATH = "best_model.keras"   # neues Keras-Format
train_dir = "data/train"
test_dir  = "data/test"

# --- Data generators (moderate Augmentation für Stabilität) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, color_mode='grayscale',
    batch_size=BATCH_SIZE, class_mode='categorical', subset='training'
)
val_gen = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, color_mode='grayscale',
    batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', shuffle=False
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, color_mode='grayscale',
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

print("\n✅ Generators ready.")
print("Train:", train_gen.samples, "| Val:", val_gen.samples, "| Test:", test_gen.samples)
print("Classes:", train_gen.class_indices)

# --- Class weights (balance) ---
y_classes = train_gen.classes
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_classes), y=y_classes)
class_weight = {i: w for i, w in enumerate(weights)}
print("Class weights:", class_weight)

# --- Model ---
def build_model(input_shape=(48,48,1), num_classes=7):
    m = Sequential([
        Conv2D(32,(3,3),activation='relu',padding='same',input_shape=input_shape),
        BatchNormalization(), MaxPooling2D(),

        Conv2D(64,(3,3),activation='relu',padding='same'),
        BatchNormalization(), MaxPooling2D(),

        Conv2D(128,(3,3),activation='relu',padding='same'),
        BatchNormalization(), MaxPooling2D(),

        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    m.compile(optimizer=Adam(learning_rate=5e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

model = build_model()
model.summary()

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)  # speichert .keras
]

# --- Train ---
steps_per_epoch = train_gen.samples // BATCH_SIZE
val_steps = max(1, val_gen.samples // BATCH_SIZE)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# --- Curves ---
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss-Verlauf'); plt.xlabel('Epoche'); plt.ylabel('Loss'); plt.legend(); plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy-Verlauf'); plt.xlabel('Epoche'); plt.ylabel('Accuracy'); plt.legend(); plt.show()

# --- Test ---
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\n✅ Test: acc={test_acc:.4f} | loss={test_loss:.4f}")
print(f"💾 Best model saved to: {MODEL_PATH}")

# --- Confusion Matrix + Report ---
pred = model.predict(test_gen, verbose=0)
y_pred = np.argmax(pred, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=labels, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7,6))
plt.imshow(cm, cmap='Blues')
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.colorbar(); plt.tight_layout(); plt.show()
