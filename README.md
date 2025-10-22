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

<img width="482" height="336" alt="Screenshot 2025-10-21 at 17 24 52" src="https://github.com/user-attachments/assets/8d515474-7041-40d7-a1c7-f0a9716efdc0" />
<img width="692" height="593" alt="Screenshot 2025-10-21 at 17 19 48" src="https://github.com/user-attachments/assets/c07b6051-ebca-45d0-80d7-5caa6c12c5f1" />
<img width="629" height="477" alt="Screenshot 2025-10-21 at 17 19 33" src="https://github.com/user-attachments/assets/3541c3be-1a81-4c9e-90cf-49c64bc59834" />
<img width="630" height="478" alt="Screenshot 2025-10-21 at 17 19 21" src="https://github.com/user-attachments/assets/d2a49a74-945d-4569-9eb7-77b06b209735" />

