# Predictive Intelligence: Building and Analysing Customer Classification Machine Learning Models
This repository contains the machine learning project for classifying customers into one of four distinct categories to optimize sales engagement strategies for the Hyper Artificial Intelligence Fund.

---

### 1. Overview
The primary business challenge is the effective and accurate classification of customers into four distinct categories of product interest (Category 0 to Category 3) using a structured dataset. The overarching objective is to support data-driven decision-making in customer engagement and, critically, to avoid outreach to customers in Category 0 who have indicated a preference not to be contacted. 
<br/>

The recall priorities for classification are:
- Category 0 = **Zero** Recall Priority
- Category 1 = **Low** Recall Priority
- Category 2 = **Medium** Recall Priority
- Category 3 = **High** Recall Priotity
Avoiding false positives in predicting Category 0 is a strategic priority to mitigate business risk and maintain trust.

The primary audience for this project is the Global Head of Sales and the sales organisation, who will use the model's predictions to inform customer targeting, customer engagement strategies, and operational risk management.

### 2. Methodology and Data
*The model was built using a structured customer dataset with 18 intial features.* <br/>
This project employed a supervised learning approach on a clearly labeled multiclass dataset. The methodology followed standard machine learning best practices:
- **Data Understanding and Cleaning:** Initial analysis using a boxplot identified significant outliers in Feature 5, which were removed using a Z-score test (absolute Z-score > 3) to prevent model bias. Only complete records (no missing values) were retained.
- **Preprocessing:** The dataset was split into a 70% training set and 30% testing set using stratified sampling to prevent information leakage and ensure representative sets. Label Encoding was used for categorical data, and Standard Scaler was selected to normalise numerical features, centering them to a mean of zero and unit variance, which is beneficial for distance-sensitive models like the Neural Network.
- **Feature Selection:** Features were selected based on a combination of Pearson correlation analysis (linear relationships) and Random Forest Classifier feature importance (non-linear relationships). The final model used seven features , with Feature 0 dropped due to its perfect correlation with Feature 4 to prevent multicollinearity.
- **Model Training:** Four classification models were trained and evaluated: Logistic Regression (baseline), K-Nearest Neighbours, Random Forest, and Neural Network.

### 3. Key Findings
**Optimal Feature Selection**<br/>
The Neural Network model is the recommended primary classification tool.<br/>
The Neural Network achieved the highest overall test accuracy ($\mathbf{88\%}$) and, most importantly, the highest Recall for Category 0 (0.90), which directly addresses the strategic business requirement to avoid misclassifying and contacting these customers. The model's balanced performance and high generalisation (7% gap between train/test accuracy) makes it suitable for deployment.

**Feature Dimensionality Reduction**<br/>
The number of features collected in the future can be reduced to improve efficiency while maintaining performance. 
- **Minimum Features for $\mathbf{> 70\%}$ Accuracy:** Four features (Features 11, 2, 8, and 6) achieved a test accuracy of $\mathbf{71.5\%}$.
- **Recommended Features for Robust Performance:** Including a fifth feature (Feature 4) significantly improves the test accuracy to $\mathbf{84.9\%}$, providing a higher confidence level for reliable classification.

### 4. Technical Stack
The following tools and libraries were used to conduct data processing, analysis and viualisation:
- **Language:** Python
- **Core Libraries:**
  * Pandas: Data manipulation and cleaning.
  * Numpy: Numerical operations (e.g. Z-score, correlation).
  * Scikit-learn: Model implementation, splitting data, feature scaling, and evaluation matrices (e.g. confusion matrices, classification reports).
  * Matplotlib: Generating boxplots, and feature importance plots.
- **Environment:** Jupyter Notebook

### 5. Future Work
Future iterations should focus on maintaining model relevance and enriching the understanding of critical customer segments.
- **Incorporate Temporal Data:** Enhance predictive performance by collecting and analysing time-series or event-driven data to capture recent behavioural shifts and seasonal patterns, moving beyond a static view of customers.
- **Implement Model Monitoring:** Establish a robust model monitoring framework and regular retraining schedules to address concept drift (the degradation of accuracy over time as customer preferences evolve).
- **Enrich Category 0 Insights:** To further reduce false positives for Category 0, collect qualitative feedback, opt-out reasons, or digital disengagement signals. This will enhance trust-based segmentation and operational risk mitigation.
- **Bias and Fairness Audits:** As the model evolves, fairness metrics and bias audits should be included in the evaluation workflow to ensure the model's decisions are fair and transparent, supporting internal stakeholder trust and auditability.
