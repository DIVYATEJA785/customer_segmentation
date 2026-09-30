import os
import numpy as np
import pandas as pd

def generate_customer_dataset(num_samples=1000, random_seed=42):
    """
    Generates a realistic synthetic customer segmentation dataset
    using Indian Rupees (₹) for Annual Income and Average Order Value (AOV).
    
    5 distinct natural behavioral profiles:
    1. High-Value Champions (High Income ₹16L-₹28L, High Spending Score 70-99, High AOV ₹5,500-₹14,000)
    2. Budget-Conscious Shoppers (Income ₹2.5L-₹6L, Low Spending Score 5-42, Low AOV ₹500-₹2,000)
    3. Young Impulse Trendsetters (Young 18-32, Income ₹4.5L-₹10L, High Spending Score 68-99, Online Spenders)
    4. Affluent Conservative Savers (Income ₹16L-₹30L, Low Spending Score 5-38, High Basket AOV ₹4,500-₹12,000)
    5. At-Risk Occasional Customers (Income ₹6L-₹14L, Moderate Spending 32-62, High Recency 110-360 days)
    """
    np.random.seed(random_seed)
    
    customers_per_cluster = [
        int(num_samples * 0.22), # High-Value Champions: ~220
        int(num_samples * 0.24), # Budget-Conscious: ~240
        int(num_samples * 0.20), # Young Impulse: ~200
        int(num_samples * 0.18), # Conservative Savers: ~180
        int(num_samples * 0.16)  # At-Risk Occasional: ~160
    ]
    # Adjust total to exactly num_samples
    diff = num_samples - sum(customers_per_cluster)
    customers_per_cluster[0] += diff

    records = []
    
    # 1. High-Value Champions (Affluent & Active Spenders)
    for _ in range(customers_per_cluster[0]):
        age = int(np.clip(np.random.normal(40, 7), 25, 62))
        gender = np.random.choice(['Female', 'Male'], p=[0.52, 0.48])
        income = int(np.clip(np.random.normal(2050000, 240000), 1550000, 2900000))
        spending_score = int(np.clip(np.random.normal(86, 6), 70, 99))
        frequency = int(np.clip(np.random.normal(28, 4), 18, 45))
        aov = round(float(np.clip(np.random.normal(8500, 1100), 5500, 14000)), 2)
        total_purchases = int(np.clip(frequency * np.random.uniform(2.8, 4.0), 55, 180))
        recency = int(np.clip(np.random.exponential(10), 1, 28))
        online = int(round(total_purchases * np.random.uniform(0.45, 0.65)))
        offline = total_purchases - online
        website_visits = int(np.clip(np.random.normal(28, 5), 15, 48))
        
        records.append({
            'Age': age, 'Gender': gender, 'Annual_Income': income,
            'Spending_Score': spending_score, 'Purchase_Frequency': frequency,
            'Average_Order_Value': aov, 'Total_Purchases': total_purchases,
            'Recency': recency, 'Online_Purchases': online,
            'Offline_Purchases': offline, 'Website_Visits': website_visits
        })

    # 2. Budget-Conscious Shoppers (Low Income, Low Spending, Moderate Recency)
    for _ in range(customers_per_cluster[1]):
        age = int(np.clip(np.random.normal(45, 10), 22, 68))
        gender = np.random.choice(['Female', 'Male'], p=[0.51, 0.49])
        income = int(np.clip(np.random.normal(420000, 80000), 220000, 650000))
        spending_score = int(np.clip(np.random.normal(24, 7), 5, 42))
        frequency = int(np.clip(np.random.normal(8, 2), 2, 14))
        aov = round(float(np.clip(np.random.normal(1200, 250), 500, 2200)), 2)
        total_purchases = int(np.clip(frequency * np.random.uniform(1.8, 2.8), 5, 38))
        recency = int(np.clip(np.random.normal(45, 15), 15, 80))
        online = int(round(total_purchases * np.random.uniform(0.20, 0.45)))
        offline = total_purchases - online
        website_visits = int(np.clip(np.random.normal(8, 2), 2, 15))
        
        records.append({
            'Age': age, 'Gender': gender, 'Annual_Income': income,
            'Spending_Score': spending_score, 'Purchase_Frequency': frequency,
            'Average_Order_Value': aov, 'Total_Purchases': total_purchases,
            'Recency': recency, 'Online_Purchases': online,
            'Offline_Purchases': offline, 'Website_Visits': website_visits
        })

    # 3. Young Impulse Trendsetters (Young, Mid Income, High Online Spenders)
    for _ in range(customers_per_cluster[2]):
        age = int(np.clip(np.random.normal(24, 3), 18, 32))
        gender = np.random.choice(['Female', 'Male'], p=[0.57, 0.43])
        income = int(np.clip(np.random.normal(720000, 110000), 450000, 1050000))
        spending_score = int(np.clip(np.random.normal(84, 6), 68, 99))
        frequency = int(np.clip(np.random.normal(20, 3), 14, 32))
        aov = round(float(np.clip(np.random.normal(2800, 450), 1600, 4500)), 2)
        total_purchases = int(np.clip(frequency * np.random.uniform(2.2, 3.2), 30, 90))
        recency = int(np.clip(np.random.exponential(12), 1, 35))
        online = int(round(total_purchases * np.random.uniform(0.72, 0.90)))
        offline = total_purchases - online
        website_visits = int(np.clip(np.random.normal(38, 6), 22, 58))
        
        records.append({
            'Age': age, 'Gender': gender, 'Annual_Income': income,
            'Spending_Score': spending_score, 'Purchase_Frequency': frequency,
            'Average_Order_Value': aov, 'Total_Purchases': total_purchases,
            'Recency': recency, 'Online_Purchases': online,
            'Offline_Purchases': offline, 'Website_Visits': website_visits
        })

    # 4. Affluent Conservative Savers (High Income, Low Spending, High Basket Value)
    for _ in range(customers_per_cluster[3]):
        age = int(np.clip(np.random.normal(54, 8), 38, 72))
        gender = np.random.choice(['Female', 'Male'], p=[0.48, 0.52])
        income = int(np.clip(np.random.normal(2200000, 260000), 1600000, 3000000))
        spending_score = int(np.clip(np.random.normal(22, 6), 5, 38))
        frequency = int(np.clip(np.random.normal(6, 2), 2, 11))
        aov = round(float(np.clip(np.random.normal(7400, 1100), 4500, 12000)), 2)
        total_purchases = int(np.clip(frequency * np.random.uniform(2.0, 3.0), 6, 32))
        recency = int(np.clip(np.random.normal(55, 16), 18, 95))
        online = int(round(total_purchases * np.random.uniform(0.35, 0.55)))
        offline = total_purchases - online
        website_visits = int(np.clip(np.random.normal(10, 3), 3, 18))
        
        records.append({
            'Age': age, 'Gender': gender, 'Annual_Income': income,
            'Spending_Score': spending_score, 'Purchase_Frequency': frequency,
            'Average_Order_Value': aov, 'Total_Purchases': total_purchases,
            'Recency': recency, 'Online_Purchases': online,
            'Offline_Purchases': offline, 'Website_Visits': website_visits
        })

    # 5. At-Risk Occasional Customers (Mid Income, Moderate Spending, Dormant / High Recency)
    for _ in range(customers_per_cluster[4]):
        age = int(np.clip(np.random.normal(46, 9), 26, 68))
        gender = np.random.choice(['Female', 'Male'], p=[0.50, 0.50])
        income = int(np.clip(np.random.normal(950000, 150000), 650000, 1350000))
        spending_score = int(np.clip(np.random.normal(48, 8), 32, 62))
        frequency = int(np.clip(np.random.normal(4, 2), 1, 9))
        aov = round(float(np.clip(np.random.normal(2400, 450), 1200, 4000)), 2)
        total_purchases = int(np.clip(frequency * np.random.uniform(1.5, 2.2), 3, 20))
        recency = int(np.clip(np.random.normal(180, 40), 110, 360))
        online = int(round(total_purchases * np.random.uniform(0.40, 0.60)))
        offline = total_purchases - online
        website_visits = int(np.clip(np.random.normal(5, 2), 1, 11))
        
        records.append({
            'Age': age, 'Gender': gender, 'Annual_Income': income,
            'Spending_Score': spending_score, 'Purchase_Frequency': frequency,
            'Average_Order_Value': aov, 'Total_Purchases': total_purchases,
            'Recency': recency, 'Online_Purchases': online,
            'Offline_Purchases': offline, 'Website_Visits': website_visits
        })

    df = pd.DataFrame(records)
    # Shuffle to mix clusters naturally
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    # Assign Customer_ID formatted as C0001, C0002...
    df.insert(0, 'Customer_ID', [f"C{i+1:04d}" for i in range(len(df))])
    
    return df

if __name__ == '__main__':
    os.makedirs('dataset', exist_ok=True)
    df = generate_customer_dataset(1000, random_seed=42)
    output_path = os.path.join('dataset', 'customers.csv')
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} customer records in INR (Rs.) at: {output_path}")
