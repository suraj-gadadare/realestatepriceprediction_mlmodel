import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define Pune localities with their characteristics
pune_localities = {
    'Koregaon Park': {'base_price': 12000, 'premium_factor': 1.8},
    'Kalyani Nagar': {'base_price': 11000, 'premium_factor': 1.7},
    'Baner': {'base_price': 9000, 'premium_factor': 1.5},
    'Wakad': {'base_price': 8500, 'premium_factor': 1.4},
    'Hinjewadi': {'base_price': 7500, 'premium_factor': 1.3},
    'Kharadi': {'base_price': 8000, 'premium_factor': 1.4},
    'Viman Nagar': {'base_price': 9500, 'premium_factor': 1.5},
    'Aundh': {'base_price': 10000, 'premium_factor': 1.6},
    'Pimpri': {'base_price': 6000, 'premium_factor': 1.1},
    'Chinchwad': {'base_price': 6500, 'premium_factor': 1.2},
    'Hadapsar': {'base_price': 7000, 'premium_factor': 1.25},
    'Kothrud': {'base_price': 8500, 'premium_factor': 1.4},
    'Warje': {'base_price': 7500, 'premium_factor': 1.3},
    'Bavdhan': {'base_price': 8000, 'premium_factor': 1.35},
    'Pashan': {'base_price': 7800, 'premium_factor': 1.32},
    'Undri': {'base_price': 6800, 'premium_factor': 1.22},
    'Kondhwa': {'base_price': 6500, 'premium_factor': 1.2},
    'Wagholi': {'base_price': 5500, 'premium_factor': 1.05},
    'Pisoli': {'base_price': 5000, 'premium_factor': 1.0},
    'Mundhwa': {'base_price': 7200, 'premium_factor': 1.28}
}

# Generate synthetic dataset
def generate_pune_real_estate_data(n_samples=2000):
    data = []
    
    for i in range(n_samples):
        # Basic property details
        locality = random.choice(list(pune_localities.keys()))
        bhk = random.choices([1, 2, 3, 4], weights=[0.25, 0.4, 0.3, 0.05])[0]
        
        # Area calculation based on BHK
        if bhk == 1:
            area = np.random.normal(600, 100)
        elif bhk == 2:
            area = np.random.normal(1000, 150)
        elif bhk == 3:
            area = np.random.normal(1400, 200)
        else:  # 4 BHK
            area = np.random.normal(1800, 250)
        
        area = max(400, area)  # Minimum area constraint
        
        # Property age
        age = random.randint(0, 25)
        
        # Floor details
        total_floors = random.randint(3, 20)
        floor = random.randint(1, total_floors)
        
        # Parking
        parking = random.choices([0, 1, 2], weights=[0.3, 0.6, 0.1])[0]
        
        # Amenities (binary features)
        gym = random.choices([0, 1], weights=[0.4, 0.6])[0]
        swimming_pool = random.choices([0, 1], weights=[0.6, 0.4])[0]
        security = random.choices([0, 1], weights=[0.2, 0.8])[0]
        garden = random.choices([0, 1], weights=[0.5, 0.5])[0]
        elevator = 1 if total_floors > 4 else random.choices([0, 1], weights=[0.3, 0.7])[0]
        
        # Furnishing status
        furnishing = random.choices(['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'], 
                                  weights=[0.4, 0.4, 0.2])[0]
        
        # Distance to nearest metro (in km)
        metro_distance = np.random.exponential(3) + 0.5
        
        # Distance to IT hubs (in km)
        it_distance = np.random.exponential(5) + 1
        
        # Calculate price based on multiple factors
        base_price = pune_localities[locality]['base_price']
        premium_factor = pune_localities[locality]['premium_factor']
        
        # Price calculation
        price_per_sqft = base_price * premium_factor
        
        # Apply various multipliers
        price_per_sqft *= (1 + bhk * 0.1)  # BHK premium
        price_per_sqft *= (1 - age * 0.015)  # Age depreciation
        price_per_sqft *= (1 + parking * 0.05)  # Parking premium
        price_per_sqft *= (1 + gym * 0.03)  # Gym premium
        price_per_sqft *= (1 + swimming_pool * 0.04)  # Pool premium
        price_per_sqft *= (1 + security * 0.02)  # Security premium
        price_per_sqft *= (1 + garden * 0.02)  # Garden premium
        price_per_sqft *= (1 + elevator * 0.03)  # Elevator premium
        
        # Furnishing premium
        if furnishing == 'Semi-Furnished':
            price_per_sqft *= 1.05
        elif furnishing == 'Fully-Furnished':
            price_per_sqft *= 1.12
        
        # Location accessibility factors
        price_per_sqft *= (1 - metro_distance * 0.01)  # Metro accessibility
        price_per_sqft *= (1 - it_distance * 0.008)  # IT hub accessibility
        
        # Floor preference (middle floors preferred)
        floor_factor = 1 + (floor / total_floors - 0.5) * 0.1
        if floor == 1:  # Ground floor discount
            floor_factor *= 0.95
        elif floor == total_floors:  # Top floor premium/discount
            floor_factor *= random.choice([0.98, 1.02])
        
        price_per_sqft *= floor_factor
        
        # Add some random noise
        price_per_sqft *= np.random.normal(1, 0.08)
        
        # Total price
        total_price = price_per_sqft * area
        
        # Price in lakhs
        price_lakhs = total_price / 100000
        
        data.append({
            'locality': locality,
            'bhk': bhk,
            'area_sqft': round(area),
            'age_years': age,
            'floor': floor,
            'total_floors': total_floors,
            'parking_spaces': parking,
            'gym': gym,
            'swimming_pool': swimming_pool,
            'security': security,
            'garden': garden,
            'elevator': elevator,
            'furnishing': furnishing,
            'metro_distance_km': round(metro_distance, 1),
            'it_distance_km': round(it_distance, 1),
            'price_lakhs': round(price_lakhs, 2)
        })
    
    return pd.DataFrame(data)

# Generate and save the dataset
if __name__ == "__main__":
    df = generate_pune_real_estate_data(2000)
    df.to_csv('pune_real_estate_data.csv', index=False)
    print(f"Generated dataset with {len(df)} samples")
    print("\nDataset overview:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nPrice statistics:")
    print(df['price_lakhs'].describe())
