# Add-to-Bag Prediction

This project tackles the **Add-to-Bag Prediction Task**, where the goal is to predict the probability that a product shown in search results will be added to a customer’s bag.

## Project Overview
Customers use the search function to find products. For each product displayed, we want to estimate the probability of it being added to the bag.  
The task uses historical interaction data, product metadata, and search query information to train machine learning models.

## Data
- **interactions_train.parquet**: User interactions between Jan 1–7, 2010. Contains labels (`viewed`, `added_to_bag`).
- **interactions_holdout_predictions.parquet**: User interactions on Jan 8, 2010. Labels not provided; requires prediction of `add_to_bag_probability`.
- **products.parquet**: Metadata for products (brand, product type, colour).

## Methodology
1. **Data Preparation & Feature Engineering**
   - Normalized IDs (product, brand, type, colour, user).
   - Added query-level features (rank normalization, query size, relative discounts).
   - Extracted time-based features (hour, day of week).
   - Computed smoothed historical add-to-bag rates for product, brand, product type, colour, and user.

2. **Modeling**
   - Trained and compared **Random Forest, LightGBM, and XGBoost**.
   - Used time-based split: Train (Jan 1–5), Validation (Jan 6–7).
   - Evaluated with probability metrics (PR-AUC, LogLoss) and ranking metrics (MAP@10, NDCG@10, HR@10).

3. **Metrics**
   - **Random Forest performed best**, delivering highest PR-AUC and MAP@10.
   - Uplift analysis showed the model increased add-to-bag rate by ~4x in the top 5% of predictions compared to the baseline average.

4. **Results**
   - Generated predictions for holdout (Jan 8).
   - Saved results in `predictions.csv` (https://github.com/vasapranavi/add_to_bag/blob/main/add_to_bag_predictions.ipynb) with required schema.

## Production Architecture
Following is the proposed production level architecture

<img width="385" height="512" alt="image" src="https://github.com/user-attachments/assets/33768a98-d474-416f-8d30-3725653f1f14" />


## Files
- `add_to_bag_predictions.ipynb`: Main notebook with preprocessing, feature engineering, modeling, evaluation, and prediction pipeline.
- `predictions.csv`: Output file with holdout predictions (search_query_time, user_id, search_query_id, product_id, rank, price_discount_percentage, add_to_bag_probability).
- `README.md`: This file.
