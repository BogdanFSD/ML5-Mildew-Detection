import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew is  fungal disease caused by Podosphaera clandestina in cherry trees."
        f" First signs may vary epending on plant species like grayish powdery patches, distorted growth twisting leaves etc.\n\n"
        f" Spores of Mildew spread by wind in warm and dry weather, spores can survive over winter in leaf piles.\n\n"
        f" First symptom is grayish spots on the upper side of leaves."
        f"\n As growth continue it could easily spread on neighbour plant \n\n"
        f" \n\n")

    st.warning(
        f"**Project Dataset**\n\n"
        f"The available dataset contains 2104 healthy leaves and 2104 infected leaves. "
        f"")

    st.success(
        f"The project has three business requirements:\n\n"
        f"1 - Visually differentiate healthy from infected cherry leaf.\n\n"
        f"2 - Prediction on a given leaf or set of leaves either infected or healthy. \n\n"
        f"3 - Download a prediction report of the examined leaves."
        )