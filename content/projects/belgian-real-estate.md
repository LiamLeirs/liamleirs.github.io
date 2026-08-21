---
title: "Belgian House Price Predictor"
draft: false
description: "An end-to-end machine learning project for collecting Belgian real-estate listings and predicting residential property asking prices."
tags:
  - Machine Learning
  - XGBoost
  - Data Science
  - Web Scraping
  - Optuna
  - SHAP
  - Streamlit
---

## Overview

The **Belgian House Price Predictor** is an end-to-end machine learning project that estimates residential property asking prices in Belgium.

Rather than starting from an existing machine learning dataset, I built the complete pipeline myself: from scraping publicly available real-estate listings, through cleaning and feature engineering, to model training, interpretation, error analysis, and deployment through an interactive application.

The final system combines **property characteristics and geographic information** with an XGBoost regression model.

The project was designed as a practical exploration of the full machine learning lifecycle rather than as a production-grade real-estate valuation system.

---

## Project Pipeline

The complete workflow consists of:

```text
Property listings
        ↓
Web scraper
        ↓
Privacy-conscious processing
        ↓
Data cleaning
        ↓
Geographic enrichment
        ↓
Exploratory data analysis
        ↓
Feature engineering
        ↓
Train / Validation / Test split
        ↓
Baseline models
        ↓
XGBoost + Optuna
        ↓
Final evaluation
        ↓
Error analysis
        ↓
Model interpretation
        ↓
Streamlit application
```

This made the project particularly useful for practicing not only model development, but also **data engineering, reproducibility, model evaluation, and deployment**.

---

## Data Collection

The dataset was collected using a custom scraper that extracts publicly available property listings from **Zimmo**.

The scraper collects attributes such as:

- asking price
- postal code
- living area
- land area
- number of bedrooms
- number of bathrooms
- construction year
- construction type
- property condition
- EPC label
- garage availability
- garden availability

Approximately **2,000 listings** were collected for the project.

### Privacy and data minimization

Because real-estate listings can contain identifying information, I deliberately minimized the amount of data retained.

Full property addresses are not stored in the modelling dataset, listing identifiers are hashed, and personal information related to sellers or agents is not retained.

Only features relevant to the machine learning task are kept.

Geographic modelling is based on generalized information such as **postal codes and postcode-derived coordinates**, rather than exact addresses.

---

## Geographic Enrichment

Location is an important determinant of property prices, but storing exact addresses was neither necessary nor desirable for this project.

Belgian postal codes were therefore enriched using external postcode data to obtain approximate:

- latitude
- longitude
- province
- postcode prefix

This allowed geographic information to be incorporated while maintaining a much coarser spatial resolution.

---

## Exploratory Data Analysis

Before training any models, I explored the dataset to better understand the Belgian housing listings and identify potential modelling problems.

The analysis focused on:

- price distributions
- living and land area
- missing data
- geographic coverage
- property characteristics
- skewed numerical variables
- unusual and extreme listings

One important observation was that the dataset was **not evenly distributed across the Belgian housing market**.

Mid-range properties were much better represented than high-value properties, while several geographic regions contained relatively few examples.

This later turned out to be an important limitation of the model.

---

## Feature Engineering

Several additional model features were created from the cleaned data.

These included:

- property age
- log-transformed living area
- log-transformed land area
- land-to-living-area ratio
- bathrooms per bedroom
- province
- postcode prefix
- latitude
- longitude

Categorical features were handled using one-hot encoding, while ordered features such as EPC label and property condition were encoded ordinally.

Missing numerical values were imputed using the median, while categorical values were imputed using their most frequent category.

The target variable was also log-transformed:

```python
y = np.log1p(price)
```

Predictions were converted back to euros using:

```python
price = np.expm1(prediction)
```

---

## Reproducible Dataset Splits

The dataset was divided into fixed:

- 70% training data
- 15% validation data
- 15% test data

The resulting splits were stored separately so that every experiment used the same properties.

This prevents accidental changes to the test set and makes comparisons between models reproducible.

The test set remained untouched during model selection and hyperparameter optimization.

---

## Baseline Models

Several regression approaches were compared before selecting the final model.

| Model            | Validation MAE | Validation RMSE | Validation R² |
| ---------------- | -------------: | --------------: | ------------: |
| Dummy Regressor  |       €171,782 |        €327,535 |        -0.066 |
| Ridge Regression |       €111,666 |        €210,448 |         0.560 |
| Random Forest    |       €103,390 |        €217,399 |         0.530 |
| XGBoost          |        €96,525 |        €188,140 |         0.648 |

XGBoost provided the strongest overall validation performance and was therefore selected for further optimization.

---

## Hyperparameter Optimization

The XGBoost model was tuned using **Optuna**.

The optimization objective was validation **Mean Absolute Error in euros**, making the optimization target directly interpretable for the problem.

Parameters such as the following were tuned:

- number of estimators
- learning rate
- tree depth
- minimum child weight
- row subsampling
- column subsampling
- L1 regularization
- L2 regularization
- gamma

The tuned model achieved a validation MAE of approximately:

**€93,044**

After selecting the best hyperparameters, the final model was retrained using the combined training and validation sets.

---

## Final Test Performance

The final model was evaluated once on the untouched test set.

| Metric |       Result |
| ------ | -----------: |
| MAE    | **€101,768** |
| RMSE   | **€179,717** |
| R²     |    **0.481** |

The model therefore explains approximately **48% of the variation in asking prices** in the test set.

An MAE of approximately €102,000 means that the predicted asking price differs from the actual asking price by around €102,000 on average.

The substantially higher RMSE indicates that a smaller number of properties produce very large prediction errors.

---

## Actual vs Predicted Prices

![Actual vs Predicted Prices](/projects/house-price-predictor/actual_vs_predicted.png)

The model captures the overall relationship between the available property characteristics and asking price.

However, prediction errors become increasingly large for expensive properties.

The predictions also become less stable toward the upper end of the housing market, where relatively few training examples are available.

---

## Error Analysis

Aggregate metrics alone do not show where a model fails, so I performed a separate error analysis on the final test predictions.

### Performance across price ranges

![Mean Absolute Error by Price Range](/projects/house-price-predictor/performance_by_price_range.png)

| Price Range | Count |      MAE | Mean Percentage Error |
| ----------- | ----: | -------: | --------------------: |
| < €250k     |    45 |  €53,657 |                 31.9% |
| €250k–€400k |   115 |  €61,520 |                 18.1% |
| €400k–€600k |    85 |  €81,852 |                 16.6% |
| €600k–€1M   |    46 | €197,762 |                 25.9% |
| > €1M       |     8 | €610,605 |                 43.6% |

The model performs most reliably within the **€250,000–€600,000 range**.

Performance deteriorates substantially above €600,000 and becomes particularly unreliable above €1 million.

The luxury segment contains only eight test observations, highlighting the limited representation of high-value properties in the available data.

---

## Residual Analysis

The model does not show a strong overall tendency to consistently overestimate or underestimate house prices.

Approximately:

- 48.8% of properties were underestimated
- 51.2% were overestimated

The mean and median residuals remained close to zero relative to the overall property-price scale.

The main issue is therefore not directional bias, but rather a smaller number of **very large prediction errors**.

---

## Geographic Performance

Performance also varies by location.

The better-represented provinces generally show broadly comparable performance, while regions with only a handful of observations produce unstable metrics.

Brussels, for example, contains only four test listings, making province-level conclusions unreliable despite its very high observed error.

Another limitation is that some listings cannot be assigned reliable geographic information, resulting in an `Unknown` province category.

These findings reinforce the need for a larger and more geographically balanced dataset.

---

## Property Characteristics

The model performs differently across several property segments.

One of the strongest patterns occurs for construction type.

| Construction Type | Count |      MAE |
| ----------------- | ----: | -------: |
| Closed            |    66 |  €55,571 |
| Semi-detached     |    70 |  €72,464 |
| Detached          |    98 | €161,990 |

Detached properties are substantially more difficult to predict.

Error also varies across condition and EPC categories, although some of these categories contain relatively few examples and are therefore harder to interpret reliably.

---

## Largest Prediction Errors

Inspection of the worst individual predictions revealed a recurring pattern.

Many of the largest errors involve properties that are:

- expensive
- detached
- unusually large
- located on very large plots
- geographically underrepresented
- partially missing information

Several properties above €1 million were heavily underestimated, while others were substantially overestimated.

This suggests that the model becomes less stable for properties that differ significantly from the majority of the training data.

---

## Model Interpretation

The final model was interpreted using both built-in XGBoost feature importance and **SHAP values**.

### Grouped Feature Importance

Geographic information is particularly important to the model.

Postcode prefix receives the largest combined importance, followed by other location-related features such as province, latitude, and longitude.

Property size is also highly influential, particularly living area and land area.

Other characteristics such as condition, EPC label, construction type, property age, and number of bathrooms provide additional predictive information.

---

## SHAP Analysis

![SHAP Summary Plot](/projects/house-price-predictor/shap_feature_importance.png)

SHAP values provide a more detailed view of both feature importance and direction.

The analysis confirms that geographic information strongly influences the predictions.

It also shows intuitive relationships for numerical features. Larger living areas and larger land areas generally push predicted prices upward, while geographic areas can create either positive or negative effects.

Because the model predicts the logarithm of price, SHAP values represent effects in log-price space rather than direct euro contributions.

The interpretation therefore focuses on **direction and relative influence** rather than claiming causal price effects.

---

## Interactive Predictor

The trained model is also exposed through a small Streamlit application.

Users can enter:

- postal code
- construction year
- living area
- land area
- bedrooms
- bathrooms
- construction type
- condition
- EPC label
- garage
- garden

The application automatically applies the same preprocessing and inference workflow used during model development:

```text
Postal code enrichment
        ↓
Feature engineering
        ↓
Preprocessing
        ↓
XGBoost
        ↓
Predicted asking price
```

![Streamlit Application](/projects/house-price-predictor/streamlit.png)

The resulting estimate is displayed as an approximate asking price.

The application is intended purely as a demonstration of the machine learning pipeline and should not be considered a professional property valuation tool.

---

## Limitations

The largest limitation of the project is **dataset size and coverage**.

Approximately 2,000 listings are sufficient to build and analyze an initial model, but several important segments of the Belgian housing market remain poorly represented.

In particular:

- luxury properties are rare
- several geographic regions contain few observations
- unusual houses can be far outside the patterns seen during training
- some listings have missing property characteristics
- precise neighbourhood-level information is unavailable

The model also predicts **advertised asking prices rather than final transaction prices**.

A property's actual sale price may differ substantially from its original listing price.

---

## What I Would Improve Next

The most important next step would be collecting a significantly larger and more balanced dataset.

In particular, I would focus on:

- increasing geographic coverage
- collecting more high-value properties
- improving representation of unusual properties
- adding richer neighbourhood-level features
- incorporating nearby amenities and accessibility
- investigating spatial validation strategies
- adding uncertainty estimates to predictions

Given the error analysis, improving the available data is likely to provide more value than further hyperparameter tuning alone.

---

## What I Learned

This project was useful because it went considerably beyond simply fitting a machine learning model.

It required thinking about the complete lifecycle of a data science project:

- collecting data rather than relying on a prepared dataset
- considering privacy during data collection
- designing a reproducible preprocessing pipeline
- separating train, validation, and test data correctly
- comparing models against meaningful baselines
- tuning without leaking information from the test set
- analyzing where a model fails instead of reporting only one metric
- interpreting a complex tree-based model
- packaging model inference separately from training
- exposing the model through a usable interface

The final model is far from a production-grade property valuation system, but that limitation is itself an important result.

The project demonstrates how **data quality, data coverage, evaluation design, and domain characteristics can matter just as much as model choice**.

---

## Technologies

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Optuna` · `SHAP` · `Streamlit` · `Matplotlib` · `Seaborn`

---

## Links

- [View the source code on GitHub](https://github.com/LiamLeirs/Belgian-House-Price-Predictor)
- [Try the interactive predictor](https://belgian-house-price-predictor-95fccevuuxyezkwapplpl4h.streamlit.app)
