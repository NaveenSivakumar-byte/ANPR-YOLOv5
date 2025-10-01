# ANPR - YOLOv5 & OCR
This project is my Final Year Project (FYP) titled “Sistem Pengecaman Plat Nombor Kenderaan dan Klasifikasi Kenderaan Menggunakan YOLOv5.” The system is designed to detect and localize Malaysian vehicle number plates and classify vehicle types (e.g., car, motorbike) using the YOLOv5 deep learning model. 

A dashboard interface is developed using Flask, enabling operators to capture vehicle images, detect number plates, recognize vehicle types, and store entry/exit history.


- YOLOv5 → Vehicle and number plate detection

- EasyOCR → Optical Character Recognition (OCR) for extracting plate numbers

- Flask → Web framework for dashboard and system integration

- MySQL → Database for storing vehicle history

- OpenCV → Image preprocessing and manipulation

- Python → Core programming language

- Bootstrap / HTML / CSS / JS → Dashboard frontend

# Demo Video
[![Watch the video](https://img.youtube.com/vi/JFLc9Kwb_yg/0.jpg)](https://www.youtube.com/watch?v=JFLc9Kwb_yg)

# Setup & Installation
1. Download Project Files
- Click the Code → Download ZIP button on this repository.
- Extract the ZIP file to your computer.

2. Create and Activate Virtual Environment
- Open a terminal inside the project folder and run:
  - python -m venv venv
  - .\venv\Scripts\activate

3. Install Dependencies
- With the virtual environment activated, run:
  - pip install -r requirements.txt

4. Run the Project
- Start the project with:
  - pyython app.py
