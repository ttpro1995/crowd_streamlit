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
    name = file.name.split(".")[0]
    input_file_path = "input/" + file.name
    input_file = open(input_file_path, 'wb')
    input_file.write(file.read())
    input_file.close()
    MODEL = "model_save/ccnn_bike.pth"
    OUTPUT_PATH = "output/"
    rand_id = str(uuid.uuid1())
    OUTPUT_NAME = OUTPUT_PATH + name

    count = count_people(input_file_path, OUTPUT_NAME, MODEL)
    st.write(count)
    output_overlay_path = OUTPUT_NAME + ".overlay.jpg"
    st.image(output_overlay_path)

    # # INPUT_NAME = "/data/my_crowd_image/dataset_batch1245/mybikedata/test_data/images/IMG_20201127_160829_821.jpg"
    # OUTPUT_PATH = "/data/my_crowd_image/dataset_batch1245/mybikedata/test_data/tmp/demo_out/"
    # rand_id = str(uuid.uuid1())
    # MODEL = "/data/save_model/adamw1_ccnnv7_t4_bike/adamw1_ccnnv7_t4_bike_checkpoint_valid_mae=-3.143752908706665.pth"
    # OUTPUT_NAME = OUTPUT_PATH + rand_id
    # input_image_file_path = OUTPUT_NAME + "_original." + ext
    # original_file = open(input_image_file_path, "wb")
    # original_file.write(file.read())
    # original_file.close()
    # count = count_people(input_image_file_path, OUTPUT_NAME, MODEL)
    # st.write(count)