# 🧠 Innovillage AI Service

Service Backend berbasis AI untuk mendeteksi motif kain songket secara otomatis. Dibangun menggunakan **FastAPI** dan **PyTorch** (MobileNetV2).

Service ini digunakan oleh aplikasi E-Commerce Innovillage untuk fitur **Smart Lens**.

## 🛠️ Tech Stack
*   **Language**: Python 3.9+
*   **Framework**: FastAPI
*   **ML Engine**: PyTorch / Torchvision
*   **Server**: Uvicorn

## 🚀 Cara Menjalankan (Lokal)

1.  **Buat Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    # venv\Scripts\activate   # Windows
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Server**
    ```bash
    uvicorn main:app --reload
    ```
    Server akan berjalan di `http://127.0.0.1:8000`.

## 📡 API Endpoints

### 1. Check Health
*   **URL**: `/`
*   **Method**: `GET`
*   **Response**: `{"status": "online", ...}`

### 2. Predict Motif
*   **URL**: `/predict`
*   **Method**: `POST`
*   **Body**: `form-data`
    *   `file`: (Binary Image File - JPG/PNG)
*   **Response**:
    ```json
    {
        "motif": "Pucuak Rabung",
        "confidence": 0.95,
        "philosophy": "..."
    }
    ```

## ☁️ Deployment (Railway)

Repository ini siap di-deploy ke [Railway](https://railway.app).

1.  **Procfile**: Sudah disertakan untuk konfigurasi start command.
2.  **Model**: File model (`songket_model.pth.zip`) disertakan di folder `model.pth/`.
3.  **Config**: Saat deploy di Railway, pastikan Root Directory diarahkan ke folder ini (jika monorepo) atau langsung root (jika repo terpisah).
