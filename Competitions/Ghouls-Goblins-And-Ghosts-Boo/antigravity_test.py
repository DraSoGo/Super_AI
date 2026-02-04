"""
Ghouls, Goblins, and Ghosts... Boo! - Kaggle Competition
Classification Task: Predict creature type from physical attributes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==================== PART 1: Load Data ====================
def load_data():
    """Load training and test datasets"""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"\nTarget distribution:\n{train_df['type'].value_counts()}")
    
    return train_df, test_df


# ==================== PART 2: EDA ====================
def explore_data(df):
    """Exploratory Data Analysis"""
    print("\n=== Dataset Info ===")
    print(df.info())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    print("\n=== Statistical Summary ===")
    print(df.describe())
    
    return df


# ==================== PART 3: Preprocessing ====================
def preprocess_data(train_df, test_df):
    """Prepare features and target"""
    # Separate features and target
    X = train_df.drop(['id', 'type'], axis=1)
    y = train_df['type']
    X_test = test_df.drop(['id'], axis=1)
    
    # Convert categorical target to numeric
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    return X_scaled, y_encoded, X_test_scaled, le


# ==================== PART 4: Model Training ====================
def train_model(X_train, y_train, X_val, y_val):
    """Train Random Forest Classifier"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\n=== Model Performance ===")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return model


# ==================== PART 5: Submission ====================
def create_submission(model, X_test, test_df, label_encoder):
    """Generate submission file"""
    predictions = model.predict(X_test)
    predictions_decoded = label_encoder.inverse_transform(predictions)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'type': predictions_decoded
    })
    
    submission.to_csv('submission.csv', index=False)
    print("\n✅ Submission file created: submission.csv")
    print(submission.head())
    
    return submission


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # 1. Load
    train_df, test_df = load_data()
    
    # 2. EDA
    explore_data(train_df)
    
    # 3. Preprocess
    X, y, X_test, le = preprocess_data(train_df, test_df)
    
    # 4. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Train
    model = train_model(X_train, y_train, X_val, y_val)
    
    # 6. Submit
    submission = create_submission(model, X_test, test_df, le)
    