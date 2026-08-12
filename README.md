# Wine-Classification-Analysis-and-Prediction

This study predicts wine type and quality using physicochemical data from two large UC Irvine repositories. 
It achieves 99.4% accuracy in classifying red vs. white wine using Logistic Regression, SVM, Decision Tree, 
and Random Forest. For wine quality classification, the system reaches an 81.1% accuracy peak, demonstrating
that Principal Component Analysis (PCA) feature selection and cross-validation pipelines significantly improve
success rates.

# 🍷Study Summary

Objective: Predict wine type (red vs. white) and grade its overall quality.
Dataset: Physicochemical features (alcohol, pH, magnesium, acids) from UCI ML Repository.
Type Accuracy: 99.4% via standard classifiers (LR, SVM, DT, RF).
Quality Accuracy: 81.1% peak utilizing PCA and cross-validation pipelines.
Evaluation: Performance visualized through normalized confusion matrix heatmaps.
Limitation: Simple models underperformed on high-quality wine variations, prompting future data refinement.
