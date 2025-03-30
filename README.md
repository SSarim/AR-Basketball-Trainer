# 🏀 Basketball AR Performance Tracker

Leverage the power of **AI/ML**, **YOLOv8**, **MediaPipe**, **NumPy**, and **OpenCV** to analyze basketball shots through real-time trajectory tracking and angle detection.

---

## 🚀 Features

- 🎯 Real-time basketball shot detection and tracking  
- 📐 Angle and trajectory analysis  
- 🤖 YOLOv8-based player and ball detection  
- 📊 Output statistics and visualizations  

---

## 📦 Setup & Installation

### ✅ Prerequisites

- Python 3.10  
- PyCharm IDE (recommended)  
- YOLOv8 pretrained model (`best.pt`)  

---

### 🔧 Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/m00nchi1d/CPS843-Final-Project.git
cd CPS843-Final-Project/CPS843_Project
```

#### 2. Open the Project in PyCharm

- Open the `CPS843_Project` folder in PyCharm.

#### 3. Configure Python Interpreter

- Navigate to `File > Settings > Project: CPS843_Project > Python Interpreter`
- Add a new interpreter using **Python 3.10 (Virtual Environment)**
- Apply changes and confirm

#### 4. (Optional) Manually Create a Virtual Environment

If not created automatically:

```bash
python -m venv venv
```

Activate the environment:

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

#### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Program

### 1. Add Your Video

Place your basketball video in the `Video/` directory.  
Supported formats: `.mp4`, `.mov`, etc.

### 2. Configure the Detection Script

Open `analysis.py`, go to **line 153**, and update this line:

```python
cap = cv2.VideoCapture('Video/[VIDEO_NAME].[format]')
```

Replace `[VIDEO_NAME]` and `[format]` accordingly.

### 3. Start the Program

Run the program via:

```bash
python main.py
```

### 4. View Results

- The video with AR overlays will appear on screen
- Output files (results/statistics) are saved to:  
  ```
  CPS843_Project/analysis
  ```

---

## 📁 Project Structure

```
CPS843-Final-Project/
├── CPS843_Project/
│   ├── analysis/                # Output data
│   ├── Basketball.v1i.yolov8    # YOLOv8 models
│   ├── runs/                    # YOLOv8 runs
│   ├── Video/                   # Input video files
│   ├── analysis.py              # Trajectory and angle detection
│   ├── gui.py                   # GUI interface (optional)
│   ├── main.py                  # Entry point
│   ├── requirements.txt         # Dependencies list
└── README.md                    # Project overview
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo  
2. Create a feature branch  
3. Commit your changes  
4. Open a pull request  

---

## 📬 Contact & Credits

Maintainers:  
- [m00nchi1d](https://github.com/m00nchi1d)  
- [SSarim](https://github.com/SSarim)  
- [shaheryar-abid](https://github.com/shaheryar-abid)  
- [p89singh](https://github.com/p89singh)  
- [nsidq](https://github.com/nsidq)  

---

## 📄 License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

🏀 _Transform your basketball game with intelligent, real-time AR analytics._
