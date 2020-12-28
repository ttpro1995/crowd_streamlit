import streamlit as st
import uuid
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from predict_single_image import count_people

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

st.write(type(file))
if file is not None:
    st.write(file.__dict__)
    ext = file.name.split(".")[-1]
    out_file = open("data/out" + "." + ext, 'wb')
    out_file.write(file.read())
    out_file.close()