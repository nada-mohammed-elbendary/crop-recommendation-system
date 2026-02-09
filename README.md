# 🌾 Crop Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-99.27%25-00C853?style=for-the-badge)

**An intelligent system that recommends the best crop to grow based on soil and climate analysis**

[Demo](#-demo) •
[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Results](#-results)

</div>

---

## 📖 Overview

An intelligent Machine Learning system that helps farmers make data-driven decisions about the most suitable crop to plant. By analyzing soil composition (N, P, K, pH) and climate conditions (temperature, humidity, rainfall), the system predicts the optimal crop with **99.27% accuracy**.

**Why This Project?**
- 🌱 Optimize agricultural productivity
- 💰 Reduce crop failure risks
- 📊 Data-driven farming decisions
- 🎯 Easy-to-use web interface

---

## ✨ Features

- ✅ **High Accuracy**: 99.27% using Random Forest Classifier
- ✅ **22 Crop Types**: Rice, wheat, cotton, coffee, and more
- ✅ **Interactive Web App**: User-friendly Streamlit interface
- ✅ **Instant Recommendations**: Real-time prediction results
- ✅ **Confidence Scores**: Displays prediction probability
- ✅ **Top 3 Alternatives**: Shows multiple crop options
- ✅ **Comprehensive Analysis**: Detailed EDA and visualizations

---

## 🎬 Demo

### Live Application Demo

**[▶️ Watch Full Demo Video](https://drive.google.com/file/d/1OnswVdfdDaPy_WMC0Onor5Va_zr72Tr0/view?usp=sharing)**

<div align="center">
  <a href="https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing">
    <img src="https://img.shields.io/badge/▶️_Demo_Video-Watch_Now-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Demo Video"/>
  </a>
</div>

### How It Works

1. **Input soil nutrients** - Adjust nitrogen, phosphorus, and potassium levels
2. **Set climate conditions** - Configure temperature, humidity, and rainfall
3. **Get instant recommendation** - Click the button to receive crop predictions
4. **View results** - See the recommended crop with confidence score and top alternatives

---

## 📊 Dataset

### Features

| Feature | Range | Unit |
|---------|-------|------|
| Nitrogen (N) | 0 - 140 | — |
| Phosphorus (P) | 0 - 145 | — |
| Potassium (K) | 0 - 205 | — |
| Temperature | 8 - 45 | °C |
| Humidity | 14 - 100 | % |
| pH | 3.5 - 10 | — |
| Rainfall | 20 - 300 | mm |

**Dataset Size**: 2,200 samples  
**Supported Crops**: 22 types  
**Balance**: Perfectly balanced (100 samples per crop)

### Supported Crops

Rice, Maize, Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

---

## 🏆 Model Performance

### Models Comparison

| Model | Test Accuracy | CV Accuracy | CV Std |
|-------|--------------|-------------|---------|
| **Random Forest** | **99.27%** | **99.09%** | **±0.65%** |
| Support Vector Machine | 98.18% | 97.82% | ±0.89% |
| K-Nearest Neighbors | 97.73% | 97.50% | ±1.02% |
| Logistic Regression | 95.45% | 95.18% | ±1.25% |

### Feature Importance

```
Nitrogen (N)     ████████████████████ 18.5%
Rainfall         ███████████████████ 17.2%
Phosphorus (P)   ██████████████████ 16.8%
Potassium (K)    █████████████████ 15.9%
Temperature      ███████████████ 14.3%
Humidity         █████████████ 12.1%
pH               ██████ 5.2%
```

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
pip package manager
```

### Installation Steps

**1. Clone Repository**
```bash
git clone https://github.com/nada-mohammed-elbendary/crop-recommendation-system.git
cd crop-recommendation-system
```

**2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Train Model**
```bash
# Open Jupyter Notebook
jupyter notebook crop_recommendation_analysis.ipynb
# Run all cells
# .pkl files will be created automatically
```

---

## 💻 Usage

### Running the App

```bash
streamlit run app.py
```

App will open at: `http://localhost:8501`

### Using Prediction Function

```python
import pickle
import numpy as np

# Load model
with open('best_crop_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Input data
input_data = np.array([[90, 42, 43, 20.9, 82.0, 6.5, 202.9]])

# Prediction
prediction = model.predict(input_data)
crop = encoder.inverse_transform(prediction)[0]

# Result
print(f"Recommended Crop: {crop}")
# Output: rice
```

---

## 📁 Project Structure

```
crop-recommendation-system/
│
├── 📊 Crop_recommendation.csv          # Dataset
├── 📓 crop_recommendation_analysis.ipynb   # Analysis notebook
├── 🎨 app.py                           # Streamlit app
├── 📋 requirements.txt                 # Dependencies
├── 📖 README.md                        # Documentation
├── 🚫 .gitignore                       # Git ignore file
│
└── 🤖 Generated after training:
    ├── best_crop_model.pkl             # Trained model
    ├── label_encoder.pkl               # Label encoder
    └── feature_names.pkl               # Feature names
```

---

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.8+ | Core programming |
| **ML Framework** | Scikit-learn | Model building |
| **Data Processing** | Pandas, NumPy | Data analysis & processing |
| **Visualization** | Matplotlib, Seaborn | Data plots |
| **Web Interface** | Streamlit | Web application |
| **Serialization** | Pickle | Model saving |

---

## 📈 Results

### Key Achievements

- ✅ **99.27% accuracy** on test data
- ✅ **99.09% accuracy** in 5-fold cross-validation
- ✅ **Low standard deviation** (±0.65%) indicating stable predictions
- ✅ **No overfitting** - consistent performance across train/test/CV
- ✅ **Balanced performance** across all 22 crop types

### Sample Predictions

| Inputs (N, P, K, T, H, pH, R) | Predicted Crop | Confidence |
|-------------------------------|----------------|------------|
| (90, 42, 43, 20.9, 82.0, 6.5, 202.9) | Rice | 99.5% |
| (20, 80, 10, 26.0, 75.0, 5.5, 120.0) | Coffee | 98.2% |
| (100, 40, 40, 25.0, 60.0, 6.0, 150.0) | Maize | 97.8% |

---

## 🔮 Future Improvements

- [ ] Add more crop types to the dataset
- [ ] Include soil type as a feature
- [ ] Integrate real-time weather API
- [ ] Add multi-language support
- [ ] Deploy on cloud platform (Heroku/AWS/Azure)
- [ ] Create mobile app version
- [ ] Add crop rotation recommendations
- [ ] Include economic factors (market prices)

---

## 📝 Important Notes

### ⚠️ Model Files

The following files are **NOT uploaded** to GitHub due to their large size:

```
❌ best_crop_model.pkl
❌ label_encoder.pkl
❌ feature_names.pkl
```

**Solution**: Run the Jupyter Notebook to generate these files

```bash
jupyter notebook crop_recommendation_analysis.ipynb
# Run all cells → Files will be created automatically
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Contact

**Nada Mohammed Elbendary**

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nada-mohammed5)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nada-mohammed-elbendary)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nadaelbendary3@gmail.com)

</div>

---

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Inspiration: Helping farmers make better agricultural decisions
- Special thanks to the open-source community

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

<div align="center">

### ⭐ If you found this project helpful, please give it a star!

**Made with ❤️ for sustainable agriculture**

---

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=nada-mohammed-elbendary.crop-recommendation-system)
![Stars](https://img.shields.io/github/stars/nada-mohammed-elbendary/crop-recommendation-system?style=social)
![Forks](https://img.shields.io/github/forks/nada-mohammed-elbendary/crop-recommendation-system?style=social)

</div>
