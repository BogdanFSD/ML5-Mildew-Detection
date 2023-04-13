import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    version = 'v2_softmax'

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    labels_distribution = plt.imread("outputs/v2_softmax/label_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")
    labels_distribution = plt.imread("outputs/v2_softmax/label_distribution_pie.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')



    st.write("### Model Performance")


    labels_distribution = plt.imread("outputs/v2_softmax/model_big_plot.png")
    st.image(labels_distribution, caption='Model History')


    st.info(
        f" 1. Loss is measure of how good model understand relationship between input and output data. val_loss is validation set and loss is training set.\n"
        f" 2. Accuracy is measure of how good model is able correctly classify input data. To be more precise it is number between correctly classified"
        f" divided by total number."
    )


    st.write("### Confusion Matrix")

    cong_mat = plt.imread("outputs/v2_softmax/confusion_matrix.png")
    st.image(cong_mat, caption='')


    st.info(
        f" There are 2 measurements in Confusion Matrix:\n\n"
        f" 1. True Positive/Negative values - represents correct prediction.\n\n" 
        f" 2. False Positive/Negative values - represents wrong prediction.\n\n"
    )


    st.write("### ROC Curve")

    roc_curve = plt.imread("outputs/v2_softmax/roccurve.png")
    st.image(roc_curve, caption='ROC Curve')


    st.info(
        f" ROC Curve evaluates how good model can define between classes. Consists of two parts:\n\n"
        f" 1. True positive rate shows how often model classify  class correctly (closer to upper left corner - better).\n\n" 
        f" 2. False positive rate shows how often model classify each class incorrectly.\n\n"
    )

    
    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(
        version), index=['Loss', 'Accuracy']))