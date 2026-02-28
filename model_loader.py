import joblib
import numpy as np
import pandas as pd

class HeartDataLoader:
    def __init__(self):
        self.scaler = joblib.load('scaler.pkl')
        self.encoder = joblib.load('encoder.pkl')
        self.model = joblib.load('rf_model.pkl')
        
        self.num_cols = list(self.scaler.feature_names_in_)
        self.cat_cols = list(self.encoder.feature_names_in_)

    def process_and_predict(self, input_df):
        df = input_df.copy()

        # 1. Numeric defaults
        for col in self.num_cols:
            if col not in df.columns:
                df[col] = 0.0

        # 2. Categorical Alignment & Safety
        for i, col in enumerate(self.cat_cols):
            known_cats = list(self.encoder.categories_[i])
            
            # If the column is missing from UI, use the first known category
            if col not in df.columns:
                df[col] = known_cats[0]
            else:
                # If the UI value is NOT in the training categories, 
                # force it to the first known category to prevent the error
                val = str(df.at[0, col])
                if val not in known_cats:
                    df.at[0, col] = known_cats[0]

        # 3. Transform
        num_scaled = self.scaler.transform(df[self.num_cols])
        cat_encoded = self.encoder.transform(df[self.cat_cols])
        
        # 4. Predict
        final_features = np.hstack([num_scaled, cat_encoded.toarray()])
        prediction = self.model.predict(final_features)[0]
        probability = self.model.predict_proba(final_features)[0][1]
        

        return prediction, probability
