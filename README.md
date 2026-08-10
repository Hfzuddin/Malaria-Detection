# Malaria Detection System

This is a web-based application designed to detect Malaria parasites in blood smear images using a Convolutional Neural Network (CNN) and Support Vector Machine (SVM) AI model.

This application uses:
- **Backend:** Python & Flask
- **Frontend:** React, TailwindCSS (via Babel Standalone & CDN)
- **AI Model:** PyTorch (Hybrid EfficientNet)

## Prerequisites
Before using this system, ensure your computer has the following installed:
1. **Python 3.8** or newer.
2. `pip` (Python package installer).

## Installation (Traditional Method)

1. **Clone the repository**
   Open your terminal/Command Prompt and run:
   ```bash
   git clone https://github.com/Hfzuddin/Malaria-Detection.git
   cd "Malaria-Detection"
   ```

2. **Install dependencies**
   It is highly recommended to use a Virtual Environment. However, you can directly install the requirements by running:
   ```bash
   pip install -r requirements.txt
   ```

## Installation (Docker Method - Recommended)

If you prefer not to manually install Python or manage libraries, you can use Docker.
Ensure **Docker Desktop** is installed on your machine.

1. Open a terminal in the project folder.
2. Run the following command:
   ```bash
   docker-compose up -d
   ```
3. The system will automatically download the operating system environment, install dependencies, and start the server. You can immediately open your browser at `http://localhost:5500`.

## Run the Application (Without Docker)

1. Navigate to the `src` directory:
   ```bash
   cd src
   ```

2. Start the Flask server:
   ```bash
   python app.py
   ```

3. Open your web browser and go to the following link:
   **http://127.0.0.1:5500**

## Usage Guide
1. On the **Dashboard** page, click **Upload Image** or **Upload Folder** to upload blood cell images (JPG/PNG format).
2. Click the **Analyze Images** button. The system will take a few seconds to process the images using the AI model.
3. Once completed, you will be redirected to the **Scanned Results** page to view whether the cell is Infected or Uninfected.
4. Previous results are automatically saved and tracked on this page.
