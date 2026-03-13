# Sepsis Detection using Machine Learning

A machine learning-based diagnostic support system for early detection and risk stratification of blood infections (sepsis) using patient vital signs and laboratory values.

## 🎯 Overview

This project implements a comprehensive ML pipeline to predict sepsis risk using clinical indicators. It provides both a command-line training interface and a web-based prediction interface for clinical use.

### Key Features
- ✅ Automated data preprocessing and feature engineering
- ✅ Multiple ML models (Random Forest & Logistic Regression)
- ✅ Comprehensive model evaluation metrics
- ✅ Interactive Streamlit web interface
- ✅ Real-time patient risk assessment
- ✅ Batch prediction capability
- ✅ Clinical decision support

## 📁 Project Structure

```
blood-infection-ml-project/
├── app/
│   └── app.py              # Streamlit interface (Frontend & Logic)
├── data/
│   └── sepsis_dataset.csv  # Raw medical data
├── models/
│   ├── infection_model.pkl # Trained model
│   └── scaler.pkl          # Feature scaler
├── notebooks/
│   └── data_analysis.ipynb # EDA and experimentation
├── src/
│   ├── data_preprocessing.py # Data cleaning & feature engineering
│   ├── train_model.py        # Model training logic
│   └── evaluation.py         # Evaluation metrics & reporting
├── requirements.txt         # Project dependencies
├── main.py                  # Training entry point
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd blood-infection-ml-project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Usage

### 1. Train Models

Run the training pipeline:

```bash
python main.py
```

This will:
- Load and preprocess data
- Split into train/validation/test sets
- Train Random Forest and Logistic Regression models
- Evaluate performance
- Save the best model and scaler

### 2. Run Streamlit App

Launch the web interface:

```bash
streamlit run app/app.py
```

Access at: `http://localhost:8501`

#### Features:
- **Manual Input**: Enter patient vitals one at a time
- **Batch Upload**: Upload CSV with multiple patients
- **Sample Data**: Test with pre-loaded examples

## 📈 Model Features

The system uses 8 clinical indicators:

| Feature | Unit | Normal Range | Clinical Significance |
|---------|------|--------------|----------------------|
| WBC Count | 10³/μL | 4.5-11.0 | Immune response |
| Temperature | °C | 36.5-37.5 | Fever detection |
| Heart Rate | bpm | 60-100 | Cardiovascular response |
| Respiratory Rate | /min | 12-20 | Pulmonary function |
| Lactate | mmol/L | 0.5-2.0 | Tissue perfusion |
| Glucose | mg/dL | 70-100 | Metabolic status |
| Platelet Count | 10³/μL | 150-400 | Coagulation status |
| Bilirubin | mg/dL | 0.1-1.2 | Hepatic function |

## 🔬 Model Performance

The models are evaluated on:
- **Accuracy**: Overall correctness
- **Precision**: False positive rate
- **Recall**: Sensitivity to sepsis cases
- **F1 Score**: Harmonic mean balance
- **ROC-AUC**: Discrimination ability
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate

## 📊 Data Pipeline

```
Raw Data → Cleaning → Feature Engineering → Scaling → 
  Split (Train/Val/Test) → Model Training → Evaluation → Deployment
```

### Preprocessing Steps
1. **Remove Duplicates**: Eliminate duplicate records
2. **Handle Missing Values**: Impute using mean/mode
3. **Remove Outliers**: Z-score based removal
4. **Feature Engineering**: Create interaction/ratio features
5. **Encoding**: Convert categorical to numeric
6. **Scaling**: StandardScaler normalization

## 🧠 Machine Learning Models

### Random Forest
- **Ensemble method** with multiple decision trees
- **Advantages**: Non-linear relationships, feature importance
- **Best for**: Complex patterns and interpretability

### Logistic Regression
- **Linear classification** model
- **Advantages**: Interpretability, probability scores
- **Best for**: Baseline and probabilistic predictions

## 📝 Example Usage

### Command Line Training
```python
from src.data_preprocessing import DataPreprocessor
from src.train_model import ModelTrainer
from src.evaluation import ModelEvaluator

# Load and preprocess
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/sepsis_dataset.csv')
X, y, features = preprocessor.preprocess(df)

# Train
trainer = ModelTrainer()
X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
model = trainer.train_random_forest(n_estimators=100)

# Evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate(model, X_test, y_test, 'RF')
```

### Web Interface Prediction
1. Open `http://localhost:8501`
2. Go to "🔍 Prediction" tab
3. Choose input method
4. Enter patient data
5. Click "🔬 Predict Risk"
6. View assessment and recommendations

## 📋 Input Data Format (CSV Upload)

When uploading multiple patients, ensure CSV has these columns:

```csv
wbc_count,temperature,heart_rate,respiratory_rate,lactate,glucose,platelet_count,bilirubin
7.5,37.2,75,16,1.5,100,250,0.8
15.2,39.5,120,28,4.5,250,80,3.2
```

## 📥 Output Format

Predictions include:
- **Risk Level**: 0 (Low Risk) or 1 (High Risk)
- **Risk Score**: Probability (0-100%)
- **Clinical Recommendations**: Suggested actions

## ⚠️ Important Disclaimers

**This tool is for educational and research purposes only.**

- ❌ Not approved for clinical decision-making without professional review
- ❌ Should complement, not replace, clinical judgment
- ❌ Always consult qualified healthcare professionals
- ❌ Not a substitute for medical diagnosis

## 🔧 Troubleshooting

### Model not found
```bash
python main.py  # Train model first
```

### Import errors
```bash
pip install -r requirements.txt  # Reinstall dependencies
```

### Streamlit not running
```bash
python -m streamlit run app/app.py
```

## 📚 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML models & metrics
- **joblib**: Model serialization
- **streamlit**: Web interface
- **matplotlib/seaborn**: Visualization

See `requirements.txt` for versions.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

Blood Infection Prediction ML Team

- Lead Developer: Revu-15
- GitHub: https://github.com/Revu-15
- Email: polamreddyrevanth.82@gmail.com
- Project Repository: blood-infection-ml-project



## 📞 Support

For issues or questions:
- Check existing documentation
- Review code comments
- Test with sample data first

## 🎓 References

- SEPSIS-3 Consensus Definitions
- Clinical Sepsis Recognition Guidelines
- Machine Learning in Healthcare Best Practices

---

**Last Updated**: 2026-03-13
