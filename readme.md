# 🔍 Real-Time Person Detection & Face Recognition

This project performs **real-time person detection** and **face recognition** using **YOLOv8**, **OpenCV**, and **face_recognition**. It takes input from an IP camera stream and draws bounding boxes and names for identified persons.

---

## 📦 Features

* Detect people in real-time using YOLOv8
* Identify known individuals using facial recognition
* Works across **Linux**, **Windows**, and **macOS**
* Compatible with IP cameras (e.g., Android IP Webcam)

---

## 🖥️ Supported Platforms

* ✅ Windows
* ✅ Linux
* ✅ macOS (with Terminal or Docker)

---

## ≥ Install Python 3 (if not already installed)

* **Windows:** Download the latest Python 3 installer from [python.org](https://www.python.org/downloads/windows/). During installation, make sure to check "Add Python to PATH".
* **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```
* **macOS:** Python 3 is usually pre-installed. You can also install it via Homebrew:
    ```bash
    brew install python3
    ```


---



<!-- ### 🐧 Linux / macOS / iOS (Terminal)

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python app.py
``` -->





## 📸 Setup IP Camera

To use an Android phone as an IP camera for this project, follow these steps:

1.  **Install IP Webcam App:**
    * On your Android device, search for and install the "IP Webcam" app by Pavel Khlebovich from the Google Play Store.

2.  **Configure and Start Stream:**
    * Open the IP Webcam app.
    * Scroll down and tap on "Start server".
    * The app will display a URL at the bottom of the screen (e.g., `http://192.168.X.X:8080`). This is your camera's stream URL. Make sure your Android device and the computer running this project are on the same local network (connected to the same Wi-Fi or wired network).

3.  **Update Project Code:**
    * Open `app.py` in your project.
    * Locate the `stream_url` variable and update it with the URL you copied from the IP Webcam app:

    ```python
    stream_url = "[http://192.168.2.191:8080/video](http://192.168.2.191:8080/video)" # Replace with your actual IP Webcam URL
    ```
    * Often, the IP Webcam app will provide a `/video` endpoint for the raw video stream, as shown in the example. If not, try the base URL first (e.g., `http://192.168.X.X:8080`).

---


### 🐧 Create and Activate a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

#### 🐧 Linux / macOS / iOS (Terminal)

```bash
# Navigate to your project directory
cd your_project_directory

# Create a virtual environment named 'env'
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

pip install -r requirements.txt
python app.py

