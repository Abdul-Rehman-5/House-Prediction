# 🏠 House Price Prediction System

End-to-end machine learning project to predict residential property prices with a deployable **Streamlit** web app.

---

## **Project Overview**
This project implements a **Linear Regression model** to predict house prices based on features such as:

- Square Footage  
- Number of Bedrooms & Bathrooms  
- Year Built  
- Lot Size  
- Garage Size  
- Neighborhood Quality  

The pipeline includes **data preprocessing**, **model training**, **evaluation**, **residual analysis**, and **feature importance extraction**. The trained model and scaler are serialized using **Joblib** for production-style inference.

---

## **Features**
- Clean **ML pipeline** with train-test split and scaling  
- **Performance metrics:** R², RMSE, MAE, MAPE  
- Residual plot and feature coefficient analysis for interpretability  
- **Streamlit app** for interactive predictions with a professional UI  
- Production-ready: separates **training** from **inference**, using `.pkl` artifacts  

---

## **Tech Stack**
- Python 3.x  
- Pandas | NumPy  
- scikit-learn  
- Matplotlib  
- Joblib  
- Streamlit  

---

## **Folder Structure**
