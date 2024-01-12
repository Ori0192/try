# import necessary libraries
import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import shap
import matplotlib
from IPython import get_ipython
from PIL import Image

# load the model and encoder object into the python file.
model = joblib.load("rta_model_deploy3.joblib")
encoder = joblib.load("ordinal_encoder2.joblib")

# set up the streamlit page settings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Accident Severity Prediction App",
                page_icon="🚧", layout="wide")
