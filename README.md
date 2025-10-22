# 🧠 Face Emotion Recognition – CNN Projekt

Datensatz von Kaggle: https://www.kaggle.com/datasets/msambare/fer2013/code?datasetId=786787&sortBy=voteCount

---

## 🎯 Ziel

Ziel ist es, Emotionen automatisch zu klassifizieren und dabei den gesamten **Machine-Learning-Prozess**
(nach dem **CRISP-DM-Modell**) umzusetzen:

- Datenaufbereitung und Augmentierung
- Training eines CNN
- Bewertung des Modells auf Testdaten
- Interpretation der Ergebnisse (z. B. Confusion Matrix, F1-Score)

---

## ⚙️ Aufbau

Das Modell besteht aus mehreren **Convolutional- und Pooling-Schichten** zur Merkmalsextraktion,
gefolgt von **Dense-Schichten** zur Klassifikation.

Zur Vermeidung von Overfitting werden verwendet:

- **Dropout**
- **Batch Normalization**
- **Klassengewichtung** bei unausgeglichenen Datensätzen

---

- Alle Bilder sind **48×48 Pixel groß** und werden als **Graustufen** eingelesen.
- 20 % der Trainingsdaten werden automatisch für die Validierung genutzt.

---

## 🚀 Training

Das Training erfolgt über Keras’ **ImageDataGenerator** mit Datenaugmentierung:

- Rotation, Verschiebung, Zoom, horizontales Flippen, Helligkeitsvariation
- **EarlyStopping**, **ReduceLROnPlateau** und **ModelCheckpoint** als Callback-Mechanismen
- Optimizer: **Adam** mit Lernrate `5e-4`
- Loss: **categorical_crossentropy**

Das beste Modell wird automatisch unter folgendem Namen gespeichert:

---

## 📊 Evaluation

Nach dem Training wird das Modell auf unabhängigen Testdaten evaluiert.
Dabei werden folgende Kennzahlen berechnet:

- **Accuracy** – Anteil der insgesamt richtigen Vorhersagen
- **Precision** – Wie viele erkannte Emotionen sind wirklich korrekt?
- **Recall** – Wie viele echte Emotionen hat das Modell erkannt?
- **F1-Score** – Ausgewogenes Maß zwischen Precision und Recall
- **Confusion Matrix** – Übersicht, welche Emotionen häufig verwechselt werden

Beispielhafte Visualisierungen:

- Verlauf von **Loss** und **Accuracy** über die Epochen
- Darstellung der **Confusion Matrix** (matplotlib)
