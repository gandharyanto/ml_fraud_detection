# Sample Fraud Detection Dataset

This document explains how to generate and use the sample fraud detection dataset for the research experiment.

## Dataset Structure

The dataset contains the following features:

- **amount**: Transaction amount (log-normal distribution)
- **time_of_day**: Hour of day (0-24)
- **day_of_week**: Day of week (0-6, where 0=Monday)
- **merchant_category**: Merchant category code (0-19)
- **transaction_type**: Type of transaction (0-4)
- **previous_failed_attempts**: Number of previous failed attempts
- **account_age_days**: Account age in days
- **transaction_frequency**: Average transaction frequency
- **balance_before**: Account balance before transaction
- **balance_after**: Account balance after transaction
- **is_foreign**: Binary flag for foreign transaction (0 or 1)
- **device_type**: Device type used (0-3)
- **ip_address_country**: IP address country code (0-49)
- **velocity_1h**: Transaction velocity in last 1 hour
- **velocity_24h**: Transaction velocity in last 24 hours
- **is_fraud**: Target variable (0=Normal, 1=Fraud)

## Generating the Dataset

### Method 1: Using the Notebook

The notebook includes a cell (Section 2) that generates the dataset. You can:
1. Run the data generation cell to create the dataset in memory
2. Optionally uncomment the save lines to export to CSV
3. Use the "Generate and Download Sample Dataset" cell to create and download a CSV file

### Method 2: Using the Python Script

Run the standalone Python script:

```bash
python generate_sample_dataset.py
```

This will create `sample_fraud_dataset.csv` with 10,000 samples by default.

To customize:
```bash
python generate_sample_dataset.py 50000 my_dataset.csv
```

This creates a dataset with 50,000 samples saved as `my_dataset.csv`.

### Method 3: In Google Colab

1. Run the data generation cell in the notebook
2. Use the "Generate and Download Sample Dataset" cell
3. Or run:
```python
from google.colab import files
df = generate_synthetic_fraud_data(n_samples=10000, fraud_ratio=0.02)
df.to_csv('sample_fraud_dataset.csv', index=False)
files.download('sample_fraud_dataset.csv')
```

## Loading the Dataset

### In the Notebook

Uncomment and use the "Alternative: Load Sample Dataset from CSV" cell:

```python
import pandas as pd
df = pd.read_csv('sample_fraud_dataset.csv')
```

### In Python Script

```python
import pandas as pd
df = pd.read_csv('sample_fraud_dataset.csv')
```

## Dataset Characteristics

- **Default size**: 10,000 samples (can be customized)
- **Fraud ratio**: 2% (200 fraud cases out of 10,000)
- **Class imbalance**: ~49:1 ratio (normal:fraud)
- **Features**: 15 features + 1 target variable
- **Data types**: Mix of continuous and discrete features

## Notes

- The dataset is synthetic and designed for academic research
- Fraud patterns are simulated with different distributions:
  - Higher transaction amounts
  - More transactions at night and weekends
  - Higher velocity (transactions per hour/day)
  - More foreign transactions
  - Newer accounts
- Random seed is set to 42 for reproducibility
- All numeric values are rounded to 2 decimal places in CSV output

## Requirements

- Python 3.6+
- numpy
- pandas

Install requirements:
```bash
pip install numpy pandas
```

