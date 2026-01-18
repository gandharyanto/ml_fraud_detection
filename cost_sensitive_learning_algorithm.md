# Algoritma Cost-Sensitive Learning untuk Fraud Detection

## Pendahuluan

**Cost-Sensitive Learning** adalah pendekatan algoritma-level untuk menangani class imbalance dengan memberikan biaya (cost) yang berbeda untuk setiap kelas selama proses training. Metode ini tidak mengubah distribusi data training, melainkan menyesuaikan fungsi cost dari algoritma pembelajaran untuk memberikan penalti yang lebih besar terhadap misklasifikasi kelas minoritas (fraud).

---

## Urutan Langkah Algoritma Cost-Sensitive Learning

### **Langkah 1: Analisis Distribusi Kelas**

**Tujuan**: Memahami tingkat ketidakseimbangan data

```python
# Menghitung jumlah sampel per kelas
class_counts = y_train.value_counts()
print(f"Class 0 (Normal): {class_counts[0]}")
print(f"Class 1 (Fraud): {class_counts[1]}")
print(f"Imbalance Ratio: {class_counts[0] / class_counts[1]:.2f}:1")
```

**Output yang diharapkan**:
- Jumlah sampel kelas mayoritas (Normal)
- Jumlah sampel kelas minoritas (Fraud)
- Rasio ketidakseimbangan

---

### **Langkah 2: Perhitungan Class Weights**

**Tujuan**: Menghitung bobot untuk setiap kelas berdasarkan distribusi data

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Mendapatkan unique classes dan memastikan terurut
classes = np.unique(y_train)

# Menghitung class weights menggunakan metode 'balanced'
class_weights = compute_class_weight(
    'balanced',
    classes=classes,
    y=y_train
)

# Membuat dictionary class weights
class_weight_dict = {
    int(classes[0]): float(class_weights[0]),  # Class 0 (Normal)
    int(classes[1]): float(class_weights[1])   # Class 1 (Fraud)
}
```

**Rumus Perhitungan**:
```
weight[i] = n_samples / (n_classes * count[i])
```

Dimana:
- `n_samples` = total jumlah sampel
- `n_classes` = jumlah kelas (2 untuk binary classification)
- `count[i]` = jumlah sampel kelas i

**Contoh Hasil**:
- Class 0 (Normal): ~0.51
- Class 1 (Fraud): ~49.99

**Penjelasan**: Kelas fraud mendapat bobot yang jauh lebih besar karena jumlahnya jauh lebih sedikit, sehingga model akan lebih "peduli" terhadap kesalahan klasifikasi fraud.

---

### **Langkah 3: Validasi dan Pembersihan Class Weights**

**Tujuan**: Memastikan format class weights sesuai dengan requirement library

```python
# Memastikan class_weight_dict adalah plain Python dict
if not isinstance(class_weight_dict, dict):
    class_weight_dict = dict(class_weight_dict)

# Memastikan semua keys dan values adalah native Python types
class_weight_dict = {
    int(k): float(v) for k, v in class_weight_dict.items()
}

# Verifikasi
print(f"Class 0 (Normal): {class_weight_dict[0]:.4f}")
print(f"Class 1 (Fraud): {class_weight_dict[1]:.4f}")
```

**Alasan**: Beberapa library (seperti TensorFlow/Keras) memerlukan format dict dengan integer keys dan float values.

---

### **Langkah 4: Implementasi pada Model RNN (LSTM, GRU, BiLSTM)**

**Tujuan**: Menerapkan class weights pada model neural network

#### **4.1. Membangun Model dengan Class Weights**

```python
def build_lstm_model(input_shape, class_weight=None):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    return model
```

#### **4.2. Training dengan Class Weights**

```python
def train_rnn_model(model, X_train, y_train, X_val, y_val, 
                    epochs=50, batch_size=128, class_weight=None):
    
    # Menggunakan class_weight yang disediakan atau default global
    if class_weight is None:
        class_weight = class_weight_dict
    
    # Validasi dan konversi class_weight
    if hasattr(class_weight, 'to_dict'):
        class_weight = class_weight.to_dict()
    
    if not isinstance(class_weight, dict):
        class_weight = dict(class_weight)
    
    # Memastikan format yang benar
    class_weight_clean = {}
    for k, v in class_weight.items():
        k_int = int(k)
        v_float = float(v)
        class_weight_clean[k_int] = v_float
    
    # Training dengan class weights
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_clean,  # <-- Cost-sensitive di sini
        verbose=1
    )
    
    return history, model
```

**Cara Kerja**:
- Setiap sampel dalam batch training dikalikan dengan bobot kelasnya
- Loss function menjadi: `weighted_loss = class_weight[class] * original_loss`
- Model akan lebih fokus meminimalkan kesalahan pada kelas dengan bobot tinggi (fraud)

---

### **Langkah 5: Implementasi pada Model Tree-Based (XGBoost, LightGBM)**

**Tujuan**: Menerapkan cost-sensitive learning pada gradient boosting models

#### **5.1. XGBoost dengan scale_pos_weight**

```python
import xgboost as xgb

# Menghitung scale_pos_weight dari class weights
scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]

# Membangun model XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # <-- Cost-sensitive di sini
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

# Training
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
```

**Rumus scale_pos_weight**:
```
scale_pos_weight = weight_class_1 / weight_class_0
                 = count_class_0 / count_class_1
```

**Cara Kerja**:
- XGBoost mengalikan gradient dan hessian dari positive class (fraud) dengan `scale_pos_weight`
- Semakin besar nilai ini, semakin besar penalti untuk miss-classify fraud

#### **5.2. LightGBM dengan scale_pos_weight**

```python
import lightgbm as lgb

# Menghitung scale_pos_weight
scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]

# Membangun model LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # <-- Cost-sensitive di sini
    random_state=42,
    verbose=-1
)

# Training
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
```

**Catatan**: LightGBM juga mendukung parameter `class_weight` atau `is_unbalance=True`, tetapi `scale_pos_weight` memberikan kontrol yang lebih presisi.

---

### **Langkah 6: Evaluasi Model**

**Tujuan**: Mengukur performa model dengan metrik yang relevan untuk imbalanced data

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

# Prediksi
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrik evaluasi
print("Model Performance:")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")  # Penting untuk fraud detection
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
```

**Metrik Prioritas untuk Fraud Detection**:
1. **Recall (Sensitivity)**: Kemampuan mendeteksi semua kasus fraud
2. **ROC-AUC**: Kemampuan membedakan fraud vs normal secara keseluruhan
3. **Precision**: Akurasi prediksi fraud (mengurangi false positive)
4. **F1-Score**: Balance antara precision dan recall

---

## Ringkasan Alur Algoritma

```
┌─────────────────────────────────────────────────────────────┐
│ 1. ANALISIS DISTRIBUSI KELAS                                │
│    - Hitung jumlah sampel per kelas                         │
│    - Tentukan rasio ketidakseimbangan                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PERHITUNGAN CLASS WEIGHTS                                │
│    - Gunakan compute_class_weight('balanced')               │
│    - Buat dictionary: {0: weight_0, 1: weight_1}            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. VALIDASI CLASS WEIGHTS                                   │
│    - Pastikan format dict dengan int keys, float values     │
│    - Verifikasi nilai weights                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│ 4. MODEL RNN     │    │ 5. MODEL TREE    │
│                  │    │                  │
│ - LSTM           │    │ - XGBoost        │
│ - GRU            │    │ - LightGBM       │
│ - BiLSTM         │    │                  │
│                  │    │                  │
│ Gunakan:         │    │ Gunakan:         │
│ class_weight=    │    │ scale_pos_weight=│
│ class_weight_dict│    │ weight_1/weight_0│
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. TRAINING MODEL                                            │
│    - Model belajar dengan weighted loss function             │
│    - Penalti lebih besar untuk misklasifikasi fraud         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. EVALUASI MODEL                                            │
│    - ROC-AUC (prioritas utama)                              │
│    - Recall (kemampuan deteksi fraud)                       │
│    - Precision, F1-Score                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Perbedaan dengan SMOTE (Data-Level Method)

| Aspek | Cost-Sensitive Learning | SMOTE |
|-------|------------------------|-------|
| **Level** | Algorithm-level | Data-level |
| **Data Training** | Original imbalanced data | Synthetic oversampled data |
| **Cara Kerja** | Menyesuaikan loss function | Menambah sampel sintetik |
| **Parameter** | `class_weight` atau `scale_pos_weight` | `sampling_strategy` |
| **Kompleksitas** | Rendah (hanya parameter) | Sedang (perlu generate data) |
| **Waktu Training** | Sama dengan baseline | Lebih lama (data lebih banyak) |

---

## Keuntungan Cost-Sensitive Learning

1. **Tidak Mengubah Data**: Bekerja dengan data asli, tidak ada risiko overfitting dari data sintetik
2. **Implementasi Sederhana**: Hanya perlu menambahkan parameter class weights
3. **Efektif untuk Imbalance**: Secara langsung menangani masalah class imbalance
4. **Interpretable**: Bobot kelas jelas menunjukkan prioritas model
5. **Fleksibel**: Dapat dikombinasikan dengan metode lain (misalnya SMOTE + Cost-Sensitive)

---

## Catatan Penting

1. **Tidak Mengubah Test Set**: Test set tetap menggunakan distribusi asli untuk evaluasi yang realistis
2. **Kombinasi dengan SMOTE**: Cost-sensitive dapat dikombinasikan dengan SMOTE untuk hasil yang lebih baik
3. **Hyperparameter Tuning**: Class weights dapat di-tune lebih lanjut jika diperlukan
4. **Domain Knowledge**: Dalam beberapa kasus, bobot dapat disesuaikan berdasarkan business cost (misalnya: biaya false negative fraud lebih mahal)

---

## Referensi Implementasi

File notebook: `fraud_detection.ipynb`
- Section: Cost-Sensitive Learning Implementation
- Cell: Class weights calculation (~line 2260)
- Cell: RNN model training (~line 2494)
- Cell: Tree-based model training (~line 4109, 4480)

---

*Dokumen ini menjelaskan implementasi Cost-Sensitive Learning untuk fraud detection berdasarkan analisis notebook `fraud_detection.ipynb`*

