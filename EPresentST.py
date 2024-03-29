import math as m
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import os
import mimetypes

# from appcopy import *
# from MainTKTest import *
from adasdad3DtestCal import *

import streamlit as st

import altair as alt
import plotly.express as px
    
import base64

st.set_page_config(
    page_title=" Deep Learning-Based Automatic RULA Assessment System ",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded")
_, titlebar, _=st.columns(3)

alt.themes.enable("dark")

st.markdown("<h1 style='text-align: center; color: white;'>AI Web Application for Automated RULA Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: grey; font-size: 16px; '>Using this artificial intelligence, you can quickly detect the worker's posture and obtain the RULA grand score.</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey; font-size: 16px; '>   </h3>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.write("")

with col2:
    st.write("")

with col3:
    st.image("Biever.jpg", width = 300, caption = '   ')
    st.markdown("<h3 style='text-align: center; color: grey; font-size: 14px;'><a href='https://indie-ct.enit.kku.ac.th' style='color: white; text-decoration: underline;'>Discover more about Indie-ct research laboratory</a></h3>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")

with col4:
    st.write("")

with col4:
    st.write("")

optionsCam = ['Intregated Camera', 'External Camera 1', 'External Camera 2', 'External Camera 3', 'External Camera 4']
CameraName = st.selectbox("Choose you camera.", optionsCam)
if CameraName == 'Intregated Camera':
    CameraName = 0
elif CameraName == 'External Camera 1':
    CameraName = 1
elif CameraName == 'External Camera 2':
    CameraName = 2
elif CameraName == 'External Camera 3':
    CameraName = 3
elif CameraName == 'External Camera 4':
    CameraName = 4

conditionMuscle2()
conditionWeight2()

st.write("")
st.write("")
st.write("")
st.write("<span style='font-weight: bold; color: rgb(255, 180, 10); font-size: 16px; '>Please check if the condtions are provided correctly before starting the application!</span>", unsafe_allow_html=True)

def simulate_loading():
    with st.spinner('Loading...'):
        # Simulate loading by waiting for a few seconds
        time.sleep(5)
    st.success('Loading complete!')

uploaded_file = st.file_uploader("Browse for a file", type=["png", "jpg", "jpeg", "mp4"], accept_multiple_files=False)
if uploaded_file is not None:
    simulate_loading()
    col1, col2, col3, col4, col5 = st.columns(5)  # Split the screen into two columns
    with col1:
        st.write("")  # Placeholder to align the button to the right
    with col2:
        # Create the 'temp' directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)

        # Save the uploaded file temporarily
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Call the pose estimation function with the file path
        image_pose_estimation2(file_path)
        # DimGraphImg(file_path)
    with col3:
        st.write("")
    with col4:
        st.write("")
    with col5:
        st.write("")

# 📹🧠🫀🏋️🤾💪♥️🩺📷🎥📽️🔎🔍🏂💪

if st.button(" 📷  Choose Live Posture Analysis using camera "):
    simulate_loading()
    col1, col2, col3, col4, col5 = st.columns(5) 
    with col1:
        st.write("")  # Placeholder to align the button to the right
    with col2:
        video_pose_estimation2(CameraName)
        # DimGraphCam(CameraName)
    with col3:
        st.write("")  # Placeholder to align the button to the right
    with col4:
        st.write("")
    with col5:
        st.write("")
