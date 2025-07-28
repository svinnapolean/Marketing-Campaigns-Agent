## 🧠 Agentic AI: Multimodel Classification Pipeline

This project explores an agentic approach to machine learning by training multiple models to predict several customer attributes simultaneously. Each model is independently trained and serialized, enabling modular deployment and inference across diverse targets.

---

### 📦 Project Structure

- `ML_market_AgentAI.ipynb`: Main notebook containing data loading, preprocessing, training, and model saving.
- `cleaned_marketing_data.csv`: Preprocessed dataset used for training.
- Saved models: Serialized `.joblib` files for each model-target pair.

---

### 🎯 Target Attributes

The pipeline predicts the following customer features:
- `Education`
- `Marital_Status`
- `Country`
- `Age_Group`

---

### 🧰 Models Used

Each target is modeled using:
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**

Models are trained independently and saved as:
```
<model_name>_<target_column>_model.joblib
```

Example:
- `logistic_regression_Education_model.joblib`
- `random_forest_Country_model.joblib`

---

### ⚙️ Workflow Summary

1. **Data Preparation**:
   - Drops non-predictive columns (`ID`, `Dt_Customer`)
   - Splits into features (`X`) and multi-target labels (`y`)
   - Train-test split with `random_state=42`

2. **Model Training**:
   - Uses `Pipeline` with `StandardScaler` for logistic regression
   - Trains each model per target column
   - Saves models using `joblib`

3. **Modularity**:
   - Easily extendable to additional targets or models
   - Supports agentic deployment where each model acts independently

---

### 🚀 Getting Started

```bash
git clone https://github.com/your-username/agentic-ai-multimodel.git
cd agentic-ai-multimodel
pip install -r requirements.txt
jupyter notebook ML_market_AgentAI.ipynb
```

---

### 📬 Contact

For questions or collaboration, reach out via [LinkedIn](https://www.linkedin.com/in/vincent-napolean-susai) or open an issue.
