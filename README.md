# CCTV Object Tracker and Summarizer

The **CCTV Object Tracker and Summarizer** is an intelligent surveillance system designed to enhance security monitoring. By leveraging state-of-the-art deep learning models, it improves object detection, tracking, and summarization in video footage. This project significantly reduces storage needs, facilitates faster analysis, and offers reliable performance for real-world surveillance environments.

---


## Introduction

Surveillance systems generate vast amounts of data, making efficient analysis challenging. Traditional motion-based detection methods struggle with accuracy in dynamic environments. The **CCTV Object Tracker and Summarizer** overcomes these limitations by implementing advanced object detection models like YOLOv8, ensuring precision, robustness, and usability.

---

## Features

- **High Accuracy**: Real-time object detection with YOLOv8.  
- **Object Tracking**: Maintains object identity across frames, even in occlusions.  
- **Event Summarization**: Generates concise video summaries highlighting key events.  
- **Custom Metrics**: Evaluates performance without relying on standard datasets.  
- **User-Friendly Interface**: Simplified interaction through a web-based UI.  

---

## System Architecture

### Design Overview

The system includes:  
1. **Video Input Handling**  
2. **Object Detection and Tracking**  
3. **Event Logging**  
4. **Video Summarization**  
5. **Streamlit-Based User Interface**  

![Architecture Diagram]("[C:\Users\ADMIN\OneDrive\Attachments\Pictures\Picture1.png](https://github.com/srajan-123/CCTV-Footage-Summariser1/blob/ef0adc467dde91e88f5a7f131a75c423af3eeb9c/Picture1.png)")  
*(Add the architecture diagram for better visualization.)*

---

## Technologies Used

- **Programming Language**: Python 3.x  
- **Libraries and Frameworks**:  
  - OpenCV (Video processing)  
  - YOLOv8 (Object detection)  
  - NumPy (Numerical computations)  
  - Streamlit (User interface)  
- **Hardware Requirements**:  
  - Multi-core CPU (GPU recommended)  
  - Minimum 8 GB RAM  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cctv-object-tracker.git
   cd cctv-object-tracker
