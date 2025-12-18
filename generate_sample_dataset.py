"""
Standalone script to generate a sample fraud detection dataset CSV file.
Run this script to create 'sample_fraud_dataset.csv'

Usage:
    python generate_sample_dataset.py
    
Requirements:
    pip install numpy pandas
"""

import numpy as np
import pandas as pd

def generate_sample_dataset(n_samples=10000, fraud_ratio=0.02, random_state=42, output_file='sample_fraud_dataset.csv'):
    """
    Generate synthetic banking transaction data with fraud labels.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples to generate
    fraud_ratio : float
        Ratio of fraudulent transactions (default 0.02 for 2% fraud)
    random_state : int
        Random seed for reproducibility
    output_file : str
        Output CSV filename
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    print(f"Generating {n_normal} normal transactions and {n_fraud} fraudulent transactions...")
    
    # Generate normal transactions
    normal_data = {
        'amount': np.random.lognormal(mean=3.5, sigma=1.2, size=n_normal),
        'time_of_day': np.random.uniform(0, 24, n_normal),
        'day_of_week': np.random.randint(0, 7, n_normal),
        'merchant_category': np.random.randint(0, 20, n_normal),
        'transaction_type': np.random.randint(0, 5, n_normal),
        'previous_failed_attempts': np.random.poisson(0.1, n_normal),
        'account_age_days': np.random.uniform(30, 3650, n_normal),
        'transaction_frequency': np.random.poisson(5, n_normal),
        'balance_before': np.random.uniform(100, 100000, n_normal),
        'balance_after': np.random.uniform(50, 100000, n_normal),
        'is_foreign': np.random.binomial(1, 0.1, n_normal),
        'device_type': np.random.randint(0, 4, n_normal),
        'ip_address_country': np.random.randint(0, 50, n_normal),
        'velocity_1h': np.random.poisson(2, n_normal),
        'velocity_24h': np.random.poisson(10, n_normal),
    }
    
    # Generate fraudulent transactions (different distributions)
    fraud_data = {
        'amount': np.random.lognormal(mean=5.0, sigma=1.5, size=n_fraud),  # Higher amounts
        'time_of_day': np.random.uniform(0, 6, n_fraud),  # More at night
        'day_of_week': np.random.choice([0, 6], n_fraud),  # More on weekends
        'merchant_category': np.random.choice([15, 16, 17, 18, 19], n_fraud),  # Specific categories
        'transaction_type': np.random.choice([3, 4], n_fraud),  # Specific types
        'previous_failed_attempts': np.random.poisson(2.5, n_fraud),  # More failed attempts
        'account_age_days': np.random.uniform(1, 180, n_fraud),  # Newer accounts
        'transaction_frequency': np.random.poisson(15, n_fraud),  # Higher frequency
        'balance_before': np.random.uniform(1000, 50000, n_fraud),
        'balance_after': np.random.uniform(0, 1000, n_fraud),  # Often drains account
        'is_foreign': np.random.binomial(1, 0.6, n_fraud),  # More foreign transactions
        'device_type': np.random.choice([0, 1], n_fraud),  # Specific devices
        'ip_address_country': np.random.choice(range(40, 50), n_fraud),  # Specific countries
        'velocity_1h': np.random.poisson(8, n_fraud),  # High velocity
        'velocity_24h': np.random.poisson(30, n_fraud),  # Very high velocity
    }
    
    # Combine data
    normal_df = pd.DataFrame(normal_data)
    normal_df['is_fraud'] = 0
    
    fraud_df = pd.DataFrame(fraud_data)
    fraud_df['is_fraud'] = 1
    
    # Combine and shuffle
    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Round numeric columns for cleaner CSV output
    numeric_cols = ['amount', 'time_of_day', 'account_age_days', 'balance_before', 'balance_after']
    df[numeric_cols] = df[numeric_cols].round(2)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Sample dataset created successfully!")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nFraud distribution:")
    print(df['is_fraud'].value_counts().to_string())
    print(f"\nFraud percentage: {df['is_fraud'].mean()*100:.2f}%")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())
    print(f"\n{'='*60}")
    
    return df

if __name__ == "__main__":
    import sys
    
    # Allow customization via command line arguments
    n_samples = 10000
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
    
    output_file = 'sample_fraud_dataset.csv'
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Generating sample dataset with {n_samples} samples...")
    df = generate_sample_dataset(n_samples=n_samples, fraud_ratio=0.02, output_file=output_file)

