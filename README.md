Automatic License Plate Recognition (ALPR) System
A real-time license plate detection and recognition system built with computer vision and deep learning. Processes both live camera feeds and video files, extracts plate numbers, and generates structured logs — all running fully locally with no external API calls or data transmission.

What It Does

Detects and highlights license plates in video frames using a trained ML model (7GB parameter set)
Recognizes plate numbers in A777AA77 format with country classification
Processes live webcam streams and pre-recorded video files
Exports timestamped logs for audit and analytics
Runs entirely offline — no cloud dependency, no data leaves the machine


Why This Matters
Most ALPR solutions are cloud-based and charge per API call. This system is:

Private by design — video and plate data never leave the local machine
Cost-effective — no per-request fees after deployment
Deployable anywhere — works without internet access
Production-ready — multithreaded video processing, GUI controls, structured output

Practical use cases include parking lot access control, warehouse entry logging, private territory security, and fleet management.

Tech Stack

Python
PyQt5 (GUI)
nomeroff-net (license plate detection and OCR)
OpenCV (video capture and frame processing)
Multithreading for non-blocking video analysis


Performance

Target processing rate: ~33 FPS
Recommended input resolution: 720p
Supports Russian-standard plates (Cyrillic and Latin character sets)
Configurable confidence threshold and minimum plate size filters


Getting Started
bashgit clone https://github.com/tiveriny/number-recognition.git
cd number-recognition
pip install -r requirements.txt
pip install nomeroff-net
python video_recognition_gui.py
Usage

Launch the app
Select input source: video file or webcam
Adjust confidence threshold if needed (default works well for 720p)
Press Start
Export report when done — logs include plate number, country, and timestamp


Project Structure
number-recognition/
├── video_recognition_gui.py   # Main application with GUI
├── video_recognition.py       # Core detection and recognition logic
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── reports/                   # Auto-generated timestamped logs
└── test_nomeroff.py           # Model validation tests

About
Built independently as a practical exploration of computer vision and applied ML. The model weights (~7GB) cover the full detection pipeline — no fine-tuning required for standard Russian plate formats.
Open to feedback, contributions, and integration questions.
