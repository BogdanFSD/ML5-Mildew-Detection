![Project look](###)

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and validation](#hypothesis-and-validation)
4. [Rationale for the model](#the-rationale-for-the-model)
5. [Trial and error](#trial-and-error)
6. [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
7. [ML Business case](#ml-business-case)
8. [Dashboard design](#dashboard-design-streamlit-app-user-interface)
9. [CRISP DM Process](#the-process-of-cross-industry-standard-process-for-data-mining)
10. [Bugs](#bugs)
11. [Deployment](#deployment)
12. [Technologies used](#technologies-used)
13. [Credits](#credits)



## App deployed hete [ml5-mildew-detection_herokuapp](LINK)

# Dataset Content

- Dataset consist of 4208 photos of cherry leaves both healthy and infected with [fungus](https://en.wikipedia.org/wiki/Powdery_mildew). Disease that affect wide range of plants however client interested in Cherry Trees mostly. All images taken from Farmy & Foods. Customer is concerned about supplying compromised quality product. 
- Dataset located at [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)
- This project based on fictious story to apply machine learning algorithms to solve problem which later on could be used in real world scenario.


## Business Requirements

Our customer Farmy & Foods contacted us to resolve non trivial issue in agricultural sector. And we are trying to create Machine Learning system that can help. Core problem is cherry leaves that are infected by fungus (Powdery Mildew). At the moment this process takes around 30 minutes per tree. Infected trees treated with fungicide. Due to size of customers (thousands trees all over the country) this process could be time consuming. In order to increase efficiency one of the solutions is Machine Learning model. Our system should make a decision based on an image of cherry leaves and give answer whether it is "Healthy" or "Infected"

- Customer interested in app that can:

1.  Visually differentiate healthy leaf from infected by powdery mildew
2.  Predicting base on image, if leaf infected or healthy


## Hypothesis and validation

1. *Hypothesis:* Leaves that are infected have marks compare to heathy ones
    - **Validation** Understand of how Powdery Mildew look like.

2. *Hypothesis:* Which model better to choose
    - **Validation** depends on the problem I solve

3. *Hypothesis:* When it perfomed better
    - **Validation** colors, filters algorithms.


### Hypothesis 1 

> Leaves that are infected have marks compare to heathy ones

If leaf is infected by Powdery Mildev we would see some classical marks as: pale yellow leaf spots, round lesions on either side which will develope to white powdery spots on the leaves. This understanding we should provide to our system. But how? Firstly, we need transform, split and basically prepare our data for learning for  best learning outcome.

Once we know we should prepare our dataset by normalization **before** training our model. In order to normalize our images we need to calculate meand and standart deviation for our images. **Mean** is dividing the sum of pixel values by the total number of pixel in dataset. **Standart deviation** basically tell us how bright or dark image is. Brighter image is more "busy" it is, if standart deviation is lov that means brightness of the picture is similar across picture. We do that with some mathematical calculation.

---
We can spot difference between healthy ad infected leaf based on this image montage.

![healthy_leaves](readme_assets/image_montage_healthy.png)
![infected_leaves](readme_assets/image_montage_powdery_mildew.png)


---


Lookig at avarage and variability images we can spot more white spotes and lines on the infected leaves


![healthy_leaves](outputs/v1/avg_var_healthy.png)
![healthy_leaves](outputs/v1/avg_var_powdery_mildew.png)


---


On the other hand no visual differenceson avarage infected and healthy leaves here

![healthy_leaves](outputs/v1/avg_diff.png)

---


System is capable of detecting differences in our leaves dataset so our learning outcome would be high. This is important step as we making sure that our model can understand patterns and features so we can make predictions for new data but with same problem.


### Hypothesis 2

---


### hypothesis 3



---


## The rationale for the model


# What I want to achieve

-
-
-
-


# Which hyperparameters I choose

- layers
- number of neurons
- kernel size
- activation function (not sigmoid)
- output
- dropout

## layers

## model creation




# Rationale to map the business requirements to the Data Visualizations and ML tasks


## Business Requirement 1: Data Visualization

- User Story


## Business Requirement 2: Classification

- User Story


## Business Requirement 3: Report
- User Story



# ML Business Case

## Powdery Mildew Detection

-
-
-
-
-



# Dashboard Design (Streamlit App User Interface)



## Page 1: Quick Project Summary

-
-
-
-


## Page 2: leaves Visualizer


-
-
-

## Page 3: Powdery mildew Detector


-
-
-
-
-

# Page 4: Project Hypothesis and Validation


-
-
--
-
-

# Page 5: ML Performance Metrics


-
-
-
-
-
-




# Deployment


1.
2.
3.
4.
5.
6.
7.
8.
9.
10.


## Forking


-
-
-
-



## Cloning repo

-
-
-
-
-
-


# Technologies used




# Credits





