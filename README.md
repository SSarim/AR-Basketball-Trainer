
# Basketball AR Performance Tracker 🏀

This project utilizes AI/ML, YOLOv8, Mediapipe, Numpy, and OpenCV for basketball trajectory tracking and angle detection. Follow the steps below to set up and run the program.

---

## Setup and Installation

### Prerequisites
- Python 3.10
- PyCharm IDE (recommended)
- YOLOv8 Pretrained Model (`best.pt`)


## Installation Steps
### 1. **Clone the Repository**
   ```bash
   git clone https://github.com/m00nchi1d/CPS843-Final-Project.git
   ```
### 2. **Open the Project**
   - Open the project folder in PyCharm IDE.
   - Open the project main folder using the following command:
   ```bash
   cd CPS843_Project
   ```

### 3. **Configure Python Interpreter**
   - Select `File` in the upper left corner.
   - Choose `Settings` > `Project: CPS843_Project` > `Python Interpreter`.
   - Click `Add Interpreter` and choose `Python 3.10 (Virtual Environment)` in the Base Interpreter section.
   - Click `Apply` and `OK` to finalize the interpreter setup.

###  Create and Activate a Virtual Environment (ONLY If the interpreter does not create it automatically)
Run the following commands in your terminal:
```bash
python -m venv venv   # Create a virtual environment
```
Then, activate the environment:
- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. **Install Requirements**
   - Run the following command in your terminal to install the necessary dependencies:
     ```bash
     pip install -r requirements.txt
     ```

---

## How to Run the Program

### 1. Upload a Video File
- Place your video file in the `CPS843_Project/Video` directory.
- Supported formats include `.mp4`, `.mov`, and more.

### 2. Edit the Detection File
- Open the `analysis.py` file.
- Edit the following line of code, on line 153:
  ```python
  cap = cv2.VideoCapture('Video/[VIDEO_NAME].[format]')
  ```
  Replace `[VIDEO_NAME]` with the name of your uploaded video file and `[format]` with the file format.

### 3. Run the Detection Program
- Run the following command in your terminal:
  ```bash
  python main.py
  ```

### 4. View the Results
- The video will be displayed live on the screen.
- At the end of processing, the results and statistics will be displayed and saved to:
  ```
  CPS843_Project/analysis
  ```
---

## **Directory Structure**
```
CPS843-Final-Project/
├── CPS843_Project/
│   ├── analysis/                # Output directory for processed videos
│   ├── Basketball.v1i.yolov8    # Yolov8 Files
│   ├── runs/                    # Yolov8 Files
│   ├── Video/                   # Directory for uploading input videos
│   ├── analysis.py              # Main detection script
│   ├── gui.py                   # GUI script  
│   ├── main.py                  # Main run script       
│   ├── requirements.txt         # List of dependencies
└── README.md                    # This README file 
```

---
## Contribution Guidelines
Contributions are welcome! If you'd like to contribute, please:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a pull request.

---

## Contact
For inquiries or support, please reach out to:
- **Project Maintainer:** m00nchi1d 
  [GitHub](https://github.com/m00nchi1d)
- **Project Maintainer:**  SSarim
  [GitHub](https://github.com/SSarim)
- **Project Maintainer:**  shaheryar-abid 
  [GitHub](https://github.com/shaheryar-abid)
- **Project Maintainer:**  p89singh 
  [GitHub](https://github.com/p89singh)
- **Project Maintainer:**  nsidq 
  [GitHub](https://github.com/nsidq)

---
## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Analyze and enhance your basketball performance with AR and AI-driven insights! 🏀


