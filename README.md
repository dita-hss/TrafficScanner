Traffic Scanner is an object detection application specifically designed to count people and vehicle traffic. It leverages the power of the YOLO (You Only Look Once) real-time object detection system to detect and classify objects.

## Overview

This project comprises two main components:

1. People Traffic Counter: This component scans an input source (like a video stream or file) and counts the number of people passing through a designated area.

2. Vehicle Traffic Counter: Similar to the People Counter, this component counts the number of vehicles passing through a specified area.

This project also includes a module called `sort.py` that's used for tracking the detected objects across multiple frames. The tracking algorithm used in `sort.py` was authored by [Alex Bewley](https://github.com/abewley).


### Prerequisites

This project has some dependencies that you need to install for it to work properly. A `requirements.txt` file is included in the repository to help facilitate this.

To install the required packages, run the following command:

pip install -r requirements.txt

Note: It's recommended to use a Python virtual environment to prevent conflicts with other projects.


### Testing

There is an included folder, `yolo-testing`, that can be used to verify that the YOLO model is functioning correctly on your device. 
