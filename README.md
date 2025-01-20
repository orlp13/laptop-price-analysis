# Analyzing Laptop Prices

## Coleta de Dados (Data Collection)
- **Sources:** 
  - Kaggle dataset: [Link to dataset](https://www.kaggle.com/datasets)
  - Manufacturer websites (e.g., Dell, HP, Lenovo)
  - Online stores (e.g., Amazon, Best Buy)

- **Tools Used:** 
  - Python with Pandas for data manipulation
  - Beautiful Soup for web scraping
  - Google Sheets for initial data collection

- **Description:** 
  The dataset contains information about various laptops, including attributes such as brand, model, processor type, RAM, storage, GPU, screen size, and price.

## Modelagem (Modeling)
- **Data Preprocessing:**
  - Handled missing values by imputing the median for numerical columns and the mode for categorical columns.
  - Encoded categorical features like brand and processor type using one-hot encoding.
  - Scaled numerical features using StandardScaler for better model performance.

- **Feature Selection:** 
  - Chose features such as `brand`, `processor type`, `RAM`, `storage`, `GPU`, `screen size`, and `price`.
  - Used correlation analysis to identify key features impacting laptop prices.

- **Algorithms:**
  - **Linear Regression:** Simple but effective model to predict prices.
  - **Random Forest:** To capture non-linear relationships and improve accuracy.
  - **Gradient Boosting:** For further enhancement and fine-tuning of the model's performance.

- **Evaluation:** 
  - Used Root Mean Squared Error (RMSE) and R-squared (R²) as evaluation metrics.
  - Cross-validation to ensure the model's robustness and generalizability.

## Conclusões (Conclusions)
- **Findings:**
  - Processor type and brand significantly affect laptop prices.
  - Laptops with higher RAM and SSD storage tend to be more expensive.
  - Gaming laptops with dedicated GPUs have a noticeable price premium.

- **Recommendations:**
  - Consumers should prioritize processor and RAM based on their use case (e.g., gaming, professional work, general use).
  - For budget-conscious buyers, considering models with HDD instead of SSD may save money, though at a performance cost.
  - Brands like Dell and Lenovo offer a balanced price-to-performance ratio.

- **Future Work:**
  - Expand the dataset to include newer laptop models and additional features like battery life.
  - Explore advanced algorithms like XGBoost or neural networks for potentially better predictions.
  - Conduct market trend analysis to predict future price movements.

## Appendix
- **Code:** 
  - Include key code snippets here or link to the full code on your GitHub repository.
  
  ```python
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LinearRegression

  # Load dataset
  data = pd.read_csv('laptop_prices.csv')

  # Data preprocessing steps
  # ...

  # Model training
  X = data[['brand', 'processor_type', 'RAM', 'storage', 'GPU', 'screen_size']]
  y = data['price']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = LinearRegression()
  model.fit(X_train, y_train)

