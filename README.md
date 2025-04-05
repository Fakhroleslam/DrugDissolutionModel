# ML for Anticancer Drug Dissolution in Supercritical CO₂

This repository contains simulation code and machine learning models for the research paper:  
**_"Towards Improved Cancer Treatment by Predicting the Solubility of Anti-Cancer Drugs in Supercritical CO₂"_**

---

## Overview

This project explores the application of machine learning to predict the solubility of anticancer drugs in supercritical carbon dioxide (SC-CO₂). A total of **22,510 models** were constructed and evaluated using a diverse set of architectures, including:

- Multilayer Perceptrons (MLP)
- Recurrent Neural Networks (LSTM, GRU)
- Gradient Boosted Trees (XGBoost)
- Hybrid and Stacked Models

The goal is to enhance **drug formulation efficiency** by delivering accurate solubility predictions based on experimental data.

---

## Machine Learning Models

The following ML architectures are implemented and optimized in this project:

- **MLP (Multilayer Perceptron)**  
  Hidden layers optimized using performance-driven search techniques.

- **LSTM (Long Short-Term Memory)**  
  A recurrent neural network variant, optimized with the Adam optimizer.

- **GRU (Gated Recurrent Unit)**  
  A simplified LSTM alternative with fewer trainable parameters.

- **XGBoost**  
  Gradient boosting decision trees with early stopping and cross-validation.

- **Hybrid & Stacked Models**  
  Ensemble models combining random forests, boosting, and neural nets for enhanced performance.

---

##  Technical Details

- **Dataset**  
  Experimental solubility data of anticancer drugs in SC-CO₂, compiled from published studies.

- **Optimization**  
  Each model underwent rigorous hyperparameter tuning for improved accuracy and generalization.

- **Validation Metrics**  
  Models were evaluated using:
  - R² (Coefficient of Determination)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)


