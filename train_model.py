import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class PuneFlatPricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.best_model_name = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Handle categorical variables
        categorical_cols = ['locality', 'furnishing']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Select features for training
        feature_cols = [
            'bhk', 'area_sqft', 'age_years', 'floor', 'total_floors',
            'parking_spaces', 'gym', 'swimming_pool', 'security', 
            'garden', 'elevator', 'metro_distance_km', 'it_distance_km',
            'locality_encoded', 'furnishing_encoded'
        ]
        
        self.feature_names = feature_cols
        X = df[feature_cols]
        y = df['price_lakhs']
        
        return X, y, df
    
    def train_models(self, X, y):
        """Train multiple models and find the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"MAE: {mae:.2f} lakhs")
            print(f"RMSE: {rmse:.2f} lakhs")
            print(f"R² Score: {r2:.4f}")
        
        # Find best model based on R² score
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")
        
        # Store all models
        self.models = {name: result['model'] for name, result in results.items()}
        
        return results, X_test, y_test
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning for the best models"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random Forest hyperparameter tuning
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        print("Tuning Random Forest...")
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        
        # Gradient Boosting hyperparameter tuning
        gb_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
        print("Tuning Gradient Boosting...")
        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_params,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        
        # Update models with best parameters
        self.models['Random Forest'] = rf_grid.best_estimator_
        self.models['Gradient Boosting'] = gb_grid.best_estimator_
        
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        print(f"Best Gradient Boosting params: {gb_grid.best_params_}")
        
        return rf_grid.best_estimator_, gb_grid.best_estimator_
    
    def get_feature_importance(self):
        """Get feature importance from the best tree-based model"""
        if self.best_model_name in ['Random Forest', 'Gradient Boosting']:
            model = self.models[self.best_model_name]
            importance = model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        return None
    
    def save_model(self, filename='pune_flat_price_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def predict(self, features):
        """Make prediction using the best model"""
        if self.best_model_name == 'Linear Regression':
            features_scaled = self.scaler.transform([features])
            prediction = self.models[self.best_model_name].predict(features_scaled)[0]
        else:
            prediction = self.models[self.best_model_name].predict([features])[0]
        
        return prediction

    def load_model(self, filename='pune_flat_price_model.pkl'):
        """Load a saved model"""
        model_data = joblib.load(filename)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        print(f"Model loaded from {filename}")

def main():
    # Check if data file exists, if not generate it
    data_file = 'pune_real_estate_data.csv'
    
    if not os.path.exists(data_file):
        print("Data file not found. Generating Pune real estate dataset...")
        try:
            # Import and run the data generation script
            import generate_pune_data
            # The data generation script should create the CSV file
            if not os.path.exists(data_file):
                print("Running data generation function...")
                df = generate_pune_data.generate_pune_real_estate_data(2000)
                df.to_csv(data_file, index=False)
                print(f"Dataset generated and saved as {data_file}")
        except ImportError:
            print("Error: generate_pune_data.py not found. Please ensure it's in the same directory.")
            return
        except Exception as e:
            print(f"Error generating data: {e}")
            return
    else:
        print(f"Using existing data file: {data_file}")
    
    # Initialize predictor
    predictor = PuneFlatPricePredictor()
    
    # Load and preprocess data
    try:
        X, y, df = predictor.load_and_preprocess_data(data_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    results, X_test, y_test = predictor.train_models(X, y)
    
    # Hyperparameter tuning
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    predictor.hyperparameter_tuning(X, y)
    
    # Get feature importance
    feature_importance = predictor.get_feature_importance()
    if feature_importance is not None:
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        print(feature_importance.head(10))
    
    # Save model
    predictor.save_model()
    
    # Create visualizations
    print("\nCreating visualizations...")
    plt.figure(figsize=(16, 12))
    
    # Price distribution
    plt.subplot(2, 3, 1)
    plt.hist(df['price_lakhs'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.title('Price Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Price (Lakhs)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    # Price by locality (top 10)
    plt.subplot(2, 3, 2)
    locality_prices = df.groupby('locality')['price_lakhs'].mean().sort_values(ascending=False)
    top_localities = locality_prices.head(10)
    bars = plt.bar(range(len(top_localities)), top_localities.values, color='lightcoral')
    plt.title('Average Price by Top 10 Localities', fontsize=12, fontweight='bold')
    plt.ylabel('Price (Lakhs)')
    plt.xticks(range(len(top_localities)), top_localities.index, rotation=45, ha='right')
    plt.tight_layout()
    
    # Price vs Area
    plt.subplot(2, 3, 3)
    plt.scatter(df['area_sqft'], df['price_lakhs'], alpha=0.6, color='green')
    plt.title('Price vs Area', fontsize=12, fontweight='bold')
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price (Lakhs)')
    plt.grid(alpha=0.3)
    
    # Price by BHK
    plt.subplot(2, 3, 4)
    df.boxplot(column='price_lakhs', by='bhk', ax=plt.gca())
    plt.title('Price Distribution by BHK', fontsize=12, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xlabel('BHK')
    plt.ylabel('Price (Lakhs)')
    
    # Feature importance
    if feature_importance is not None:
        plt.subplot(2, 3, 5)
        top_features = feature_importance.head(8)
        bars = plt.barh(top_features['feature'], top_features['importance'], color='orange')
        plt.title('Top Feature Importances', fontsize=12, fontweight='bold')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
    
    # Model comparison
    plt.subplot(2, 3, 6)
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    colors = ['red', 'blue', 'purple']
    bars = plt.bar(model_names, r2_scores, color=colors)
    plt.title('Model Performance (R² Score)', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Dataset size: {len(df)} samples")
    print(f"Number of features: {len(predictor.feature_names)}")
    print(f"Best model: {predictor.best_model_name}")
    print(f"Best R² score: {results[predictor.best_model_name]['r2']:.4f}")
    print(f"Best RMSE: {results[predictor.best_model_name]['rmse']:.2f} lakhs")
    print(f"Best MAE: {results[predictor.best_model_name]['mae']:.2f} lakhs")
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    # Create a sample property for prediction
    sample_locality = 'Baner'
    sample_furnishing = 'Semi-Furnished'
    
    # Encode categorical variables
    locality_encoded = predictor.label_encoders['locality'].transform([sample_locality])[0]
    furnishing_encoded = predictor.label_encoders['furnishing'].transform([sample_furnishing])[0]
    
    sample_features = [
        3,  # bhk
        1200,  # area_sqft
        5,  # age_years
        7,  # floor
        15,  # total_floors
        1,  # parking_spaces
        1,  # gym
        0,  # swimming_pool
        1,  # security
        1,  # garden
        1,  # elevator
        2.5,  # metro_distance_km
        5.0,  # it_distance_km
        locality_encoded,  # locality_encoded
        furnishing_encoded  # furnishing_encoded
    ]
    
    predicted_price = predictor.predict(sample_features)
    
    print(f"Sample Property:")
    print(f"  Location: {sample_locality}")
    print(f"  BHK: 3")
    print(f"  Area: 1200 sq ft")
    print(f"  Age: 5 years")
    print(f"  Floor: 7/15")
    print(f"  Furnishing: {sample_furnishing}")
    print(f"  Predicted Price: ₹{predicted_price:.2f} lakhs")
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()