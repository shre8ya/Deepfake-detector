# 🎭 Deepfake Detector

A Flask-based web application that detects deepfake videos using AI.

---

## 🚀 Features
- Upload video files
- Face detection using OpenCV
- AI-based classification
- Real-time processing

---

## 🛠️ Tech Stack
- Python
- Flask
- OpenCV
- PyTorch / Transformers

---

## 📁 Project Structure

deepfake-detector/
│
├── src/
│   ├── app.py
│   ├── detector.py
│   ├── model.py
│
├── templates/
│   └── index.html
│
├── uploads/
├── venv/

---

## ▶️ How to Run

```bash
# Clone repo
git clone https://github.com/your-username/deepfake-detector.git

# Go into folder
cd deepfake-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
python src/app.py