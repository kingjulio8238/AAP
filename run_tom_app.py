#!/usr/bin/env python3
import os
import sys

# Create a virtual environment with the right dependencies
print("Installing required packages...")
os.system('python -m pip install numpy==1.23.5 opencv-python==4.5.5.64 --force-reinstall')
os.system('python -m pip install nvidia-ml-py3')
os.system('python -m pip install pynvml')

# Run the app in a new Python process to avoid import conflicts
print("Starting ToM Generator app...")
os.system('python tom_app.py') 