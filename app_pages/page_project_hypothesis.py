import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")


    st.write("### Hypotesis 1 ")

    st.success(
        f"Leaves that are infected have marks compare to heathy ones."
    )
    st.info(
        f"According to research infected leaves should have marks like grayish spots, tiny, round, black fungal structures (cleistothecia)."
        f"Infected surface dradually spreads and covers entire leaf which give a look of dusted leaf with white powder"
        f"These symptoms are sufficient to determine if leaf is infected"
    )
     
    st.warning(
        f" Image montage reveals defference between healthy and infected leaves."
        f" Model been able to make correct prediction in 97% of cases."
        f" Image variability showed color difference mostly in the center of the leaves however no clear patterns identified"
    )

    st.write("### Hypotesis 2 ")

    st.success(
        f"Image that contains overlap or partial image of the leaf."
    )

    st.info(
        f" Take images of each leaf might be not the most time efficient way of collecting data."
        f" One of the solution is take a photo of a tree that we are interested in. However images of leaves might be not so clear or good quality."
        f" Infected leaves could be overlaped by healthy and vice-versa or regular branch could be in front.\n\n"
        f" We cut images in 3 equal parts to mimic real world situation. "
    )

    st.warning(
        f" As leaves been cropped it is much more difficult for models to pich up patterns by which they can make prediction."
        f" Our model tend to overfit and made bad predictions on a new dataset of croped leaves."
    )

    image_model_cut = Image.open('outputs/v2_cut/model_big_plot.png')
    st.image(image_model_cut, caption='Cut images loss/accuracy perfomance')

    st.write("### Hypotesis 3 ")

    st.success(
        f"Which activation functions is better perfomed at this model."
    )

    st.info(
        f" Each ot he functions popular for image classification tasks. Sigmoid function used when we have just 2 classes while softmax 2 and more."
        f" To choose which function is better we will train model with each of them and than compare results."
    )

    st.warning(
        f" Model that use softmax outperformed sigmoid function."
    )

    image_model_func_1 = Image.open('outputs/v1/model_big_plot.png')
    st.image(image_model_func_1, caption='Loss/Accuracy Sigmoid plot')


    image_model_func_2 = Image.open('outputs/v2_softmax/model_big_plot.png')
    st.image(image_model_func_2, caption='Loss/Accuracy Softmax plot')