# Player Injury Risk Monitoring System (PIRMS)

PIRMS is a synthetic-data-driven machine learning system designed to model and analyze NFL-style player injury risk. It includes a reproducible data pipeline, exploratory data analysis (EDA), and predictive modeling components using logistic regression and random forest classifiers.

This project was created as part of a graduate-level data product development course. Because real NFL injury data is not publicly available in full detail, this implementation uses a fully synthetic dataset that mimics real-world structure and workload patterns.

---

## 📌 Features

### **Data Generation**
- Synthetic player roster generator  
- Weekly workload statistics  
- Injury probability simulation based on position  
- Automatic merging and preprocessing  

### **Feature Engineering**
- Touches, workload, cumulative injury history  
- Rolling averages  
- Games played/missed (recent windows)  

### **Exploratory Data Analysis**
Automatically generates plots:
- Missing value heatmap  
- Histograms  
- Correlation matrix  
- Injuries by position  
- Workload vs injury boxplot  

### **Machine Learning Models**
- Logistic Regression  
- Random Forest  
- ROC curves  
- Feature importance  
- Classification reports  
All results are saved in `/models/`.

---

## 📁 Project Structure
pirms_project_root/
│
├── requirements.txt
├── README.md
│
└── pirms_project/
├── main.py
├── pipeline.py
├── config.py
├── data_loader.py
├── preprocess.py
├── feature_engineering.py
├── visualizations.py
├── ml_models.py

---

## 🚀 How to Run Locally

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run the full pipeline
python pirms_project/main.py


The system will:

1. Generate a synthetic dataset  
2. Run EDA  
3. Train ML models  
4. Save reports and plots into:
/plots
/models
pirms_synthetic_dataset.csv

---

## 🌩 Deployment Notes

This project can be deployed to any Python-capable cloud service including:

- Render.com  
- Azure Web Apps  
- AWS Elastic Beanstalk  
- Google Cloud Run  
- Replit or GitHub Codespaces  

#### Requirements for deployment:
- `requirements.txt` must be at project root  
- Entry point should call:  
python pirms_project/main.py
- For web deployment, wrap the output in a Flask or FastAPI layer  
(placeholder for future expansion)

---

## 🔮 Future Work

- Replace synthetic data with real-world NFL injury datasets  
- Build interactive dashboards with Plotly or Streamlit  
- Add live API endpoints for real-time risk scoring  
- Containerize with Docker for easier cloud deployment  

---

## 👤 Author

Sean MacKelvey  
Graduate Student, Data Science  
Player Injury Risk Monitoring System (PIRMS)
