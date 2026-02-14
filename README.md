# Handwriting OCR Addition Calculator

A Python web application that extracts numbers from handwritten addition problems (e.g., `23 + 45`) using PaddleOCR and computes the sum.

## Features
- **Upload Handwriting**: Supports images of handwritten addition problems.
- **Smart Filtering**: Ignores faint background text and noise.
- **Dark Mode UI**: Modern, glassmorphism-based interface.
- **Production Ready**: Includes Gunicorn and Docker setup.

## Deployment with Docker

The easiest way to deploy this application is using **Docker**. This ensures all dependencies (OCR libraries, system packages) are correctly installed.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your machine.
- [Git](https://git-scm.com/) (optional, to clone the repo).

### Step 1: Build Image
Run the following command in the project root:
```bash
docker build -t handwriting-ocr .
```

### Step 2: Run Container
Run the container on port 5000:
```bash
docker run -p 5000:5000 handwriting-ocr
```

Access the application at `http://localhost:5000`.

### Manual Deployment (without Docker)

If you prefer to run it directly on a server (e.g., EC2, VPS):

1. **Install System Dependencies (Ubuntu/Debian)**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip libgomp1 libgl1 libglib2.0-0
   ```

2. **Install Python Packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run with Gunicorn**:
   ```bash
   gunicorn --bind 0.0.0.0:5000 app:app
   ```
