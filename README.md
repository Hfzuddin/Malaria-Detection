# Malaria Detection System

A web-based AI application designed to accurately detect and classify Malaria parasites in blood smear images using a custom Hybrid EfficientNet model. 

## 🌟 Features
- **AI-Powered Analysis**: Utilizes a PyTorch-based Hybrid EfficientNet architecture to achieve high accuracy in detecting infected cells.
- **Three-Class Classification**: Identifies cells as `INFECTED`, `UNINFECTED`, or `NON_BLOOD`.
- **Fast and Accurate**: Processes both individual cell images and full blood slide scans.
- **Visual Highlighting**: Automatically highlights the detected parasites in the image using computer vision techniques (OpenCV).
- **Responsive UI**: A modern dashboard built with React and TailwindCSS.

## 🛠️ Technology Stack
- **Backend:** Python, Flask, Werkzeug
- **AI Model:** PyTorch (Hybrid EfficientNet_B3), Scikit-Learn (SVM)
- **Computer Vision:** OpenCV (`opencv-python`)
- **Frontend:** React, TailwindCSS (CDN)

## 📋 Prerequisites
- **Git** (for cloning)
- **Docker Desktop** (Recommended)
- *OR* **Python 3.9+** (if running manually)

---

## 🚀 Getting Started

You can run this application either using **Docker (Recommended)** or by running it manually using Python.

### Option 1: Using Docker (Recommended & Easiest)
Using Docker guarantees that all dependencies (including heavy libraries like OpenCV and PyTorch) are correctly installed without messing up your local environment.

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hfzuddin/Malaria-Detection.git
   cd Malaria-Detection
   ```

2. **Start Docker Desktop** on your computer.

3. **Run the Application**
   Open your terminal in the project directory and run:
   ```bash
   docker-compose up -d --build
   ```
   *Note: The first time you run this, it may take several minutes to download the Python image and install the AI libraries.*

4. **Access the Web App**
   Open your browser and navigate to: [http://localhost:5500](http://localhost:5500)

5. **Stop the Application**
   ```bash
   docker-compose down
   ```

### Option 2: Traditional Manual Installation (Without Docker)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hfzuddin/Malaria-Detection.git
   cd Malaria-Detection
   ```

2. **Create a Virtual Environment (Highly Recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   cd src
   python app.py
   ```

5. **Access the Web App**
   Open your browser and navigate to: [http://localhost:5500](http://localhost:5500)

---

## 💡 How to Use
1. On the **Dashboard**, click **Upload Image** or **Upload Folder** to provide blood smear images (JPG/PNG).
2. Click **Analyze Images**. The AI model will process each image.
3. The results page will display the diagnosis:
   - **INFECTED** (Malaria parasite detected - visual highlights will be added).
   - **UNINFECTED** (Healthy cell).
   - **NON_BLOOD** (Image is not recognized as a valid blood cell).

