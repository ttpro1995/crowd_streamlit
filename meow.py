import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

st.write(type(file))
if file is not None:
    st.write(file.__dict__)

    bytes_data = file.read()
    # st.write(bytes_data)
    st.image(bytes_data, use_column_width=True)

