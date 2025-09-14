# Pune Real Estate Price Prediction App

This project is a **Machine Learning-powered web application** that predicts Pune flat prices based on various features such as location, size, and amenities. It uses trained models (Random Forest, Gradient Boosting, Linear Regression) and a Streamlit frontend.

## 🚀 Features
- User-friendly web interface to input flat features.
- Predict Pune flat prices instantly.
- Supports multiple machine learning models (Random Forest, Gradient Boosting, Linear Regression).
- Model saved with Joblib for fast loading.
- Built with Python, Pandas, Scikit-learn, and Streamlit.

## 🗂 Project Structure
├── app.py # Streamlit frontend
├── pune_flat_price_model.pkl # Saved ML model
├── requirements.txt # Python dependencies
├── data/ # Dataset(s)
├── notebooks/ # Jupyter notebooks for training & EDA
├── utils/ # Helper functions (preprocessing, feature engineering)
└── README.md # Project documentation

bash
Copy code

## ⚙️ Installation & Setup
```bash
git clone https://github.com/your-username/pune-real-estate-price-prediction.git
cd pune-real-estate-price-prediction
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
📁 Models
The pune_flat_price_model.pkl file contains the trained model. Replace it with your own retrained model if required.

🛠 Utilities
The utils/ folder contains helper scripts for:

Data preprocessing

Feature engineering

Model loading

🐳 Run with Docker
bash
Copy code
docker build -t pune-real-estate .
docker run -p 8501:8501 pune-real-estate
Visit http://localhost:8501 to use the app.

##🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

✨ Future Enhancements
Add location-based visualizations on map

Support more ML models & hyperparameter tuning

Deployment on cloud (AWS/GCP/Azure)
