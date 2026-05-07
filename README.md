# 🩺 Diabetic Retinopathy Detection & Explainable AI System

## 📌 Overview

This project is a deep learning-based system for detecting **Diabetic
Retinopathy (DR)** from retinal fundus images.\
It classifies images into **5 severity levels** and integrates
**Explainable AI (Grad-CAM)** to provide visual interpretations of
predictions.

------------------------------------------------------------------------

## 🎯 Objectives

-   Detect Diabetic Retinopathy severity (5 classes)
-   Achieve high performance using EfficientNet
-   Provide model interpretability using Grad-CAM
-   Build a user-friendly web interface
-   Generate automated medical-style reports

------------------------------------------------------------------------

## 🧠 Key Features

-   5-class classification (No DR → Proliferative DR)
-   EfficientNetB3 deep learning model
-   Advanced preprocessing (Ben Graham method + circular cropping)
-   Explainable AI using Grad-CAM
-   Real-time prediction via Streamlit
-   Multi-image upload support
-   Clinical interpretation of results
-   Automated PDF report generation

------------------------------------------------------------------------

## 📊 Model Performance

-   Training Accuracy: \~79%
-   Validation Accuracy: \~81%
-   Training Loss: \~0.072
-   Validation Loss: \~0.065

------------------------------------------------------------------------

## 🧠 Explainable AI (XAI)

We used Grad-CAM to generate heatmaps highlighting important regions in
the retinal image.

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is for educational purposes only and not a medical
diagnosis tool.
