# Formative 3 – Mathematics for Machine Learning

This project demonstrates fundamental mathematical concepts essential for machine learning through practical implementations and visualizations. We build probability distributions from scratch, apply Bayesian inference to real-world data, and implement gradient descent for optimization.

## 📚 Project Overview

This repository contains three comprehensive modules exploring:
- **Bivariate Normal Distributions** implemented from mathematical first principles
- **Bayesian Probability** applied to sentiment analysis
- **Gradient Descent** optimization for linear regression

## 📂 Project Structure

```
Formative3_MathforML/
├── README.md
├── part1/
│   ├── Probability_Distributions.ipynb
│   └── medical_insurance_2026_kaggle.csv
├── part2/
│   └── Probability_by_bayes_theorem (1).ipynb
└── part4/
    └── gradientDescent_calculation.ipynb
```

## 📘 Part 1: Probability Distributions

**File:** [part1/Probability_Distributions.ipynb](part1/Probability_Distributions.ipynb)

**Dataset:** Medical Insurance 2026 (Kaggle)

### What It Does
Implements the **bivariate normal probability density function** from scratch without using statistical libraries. Analyzes the relationship between age and BMI in medical insurance data.

### Key Features
- Custom implementation of bivariate normal PDF formula
- Parameter estimation (means, standard deviations, correlation coefficient)
- Visualization of joint probability distributions
- Contour plots showing probability density across two dimensions

### Mathematical Concepts
- Bivariate normal distribution
- Covariance and correlation
- Probability density functions
- Statistical parameter estimation

### Outputs
- Scatter plots of age vs BMI
- 3D surface plots of probability density
- Contour plots showing equiprobability curves
- Calculated statistics: μ_x, μ_y, σ_x, σ_y, ρ

---

## 📗 Part 2: Bayesian Probability

**File:** [part2/Probability_by_bayes_theorem (1).ipynb](part2/Probability_by_bayes_theorem%20(1).ipynb)

**Dataset:** IMDb Movie Reviews

### What It Does
Applies **Bayes' Theorem** to calculate the probability that a review is positive given the presence of specific keywords. Demonstrates probabilistic reasoning in sentiment analysis.

### Key Features
- Keyword-based sentiment analysis (positive: "excellent", "wonderful", "amazing", "love")
- Implementation of Bayes' theorem from scratch
- Likelihood, prior, and posterior probability calculations
- Marginal probability computation

### Mathematical Concepts
- Bayes' Theorem: P(Positive | Keyword) = P(Keyword | Positive) × P(Positive) / P(Keyword)
- Prior probability P(Positive)
- Likelihood P(Keyword | Positive)
- Marginal probability P(Keyword)
- Posterior probability P(Positive | Keyword)

### Functions Implemented
- `keyword_count()`: Count occurrences of keywords in reviews
- `likelihood()`: Compute P(Keyword | Positive)
- `marginal()`: Compute P(Keyword)
- `posterior()`: Compute P(Positive | Keyword) using Bayes' theorem

### Outputs
- Prior probability of positive reviews
- Likelihood values for each keyword
- Posterior probabilities showing keyword-sentiment correlations

---

## 📙 Part 4: Gradient Descent

**File:** [part4/gradientDescent_calculation.ipynb](part4/gradientDescent_calculation.ipynb)

**Problem:** Linear Regression on simple dataset

### What It Does
Implements **gradient descent optimization** from scratch to find optimal parameters for linear regression. Demonstrates manual computation of gradients and iterative parameter updates.

### Key Features
- Manual gradient computation from first principles
- Step-by-step parameter updates
- Visualization of learning progression
- MSE (Mean Squared Error) tracking

### Mathematical Concepts
- Linear regression model: y = mx + b
- Cost function: J(m,b) = (1/n) Σ(y - ŷ)²
- Gradient computation:
  - ∂J/∂m = (-2/n) Σ x(y - ŷ)
  - ∂J/∂b = (-2/n) Σ(y - ŷ)
- Parameter update rule:
  - m_new = m - α × ∂J/∂m
  - b_new = b - α × ∂J/∂b

### Functions Implemented
- `predict()`: Compute predictions for given parameters
- `compute_mse()`: Calculate Mean Squared Error
- `compute_gradients()`: Derive gradients of MSE
- `update_parameters()`: Apply gradient descent update rule
- `gradient_descent()`: Full iterative optimization procedure

### Hyperparameters
- Learning rate (α): 0.1
- Iterations: 3
- Initial values: m = -1, b = 1
- Training data: X = [1, 3], Y = [3, 6]

### Outputs
- Step-by-step iteration logs showing predictions, MSE, gradients, and parameter updates
- Convergence plots for m (slope) and b (intercept)
- Error evolution graph showing MSE reduction

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib scipy
```

### Running the Notebooks

1. **For Part 1 (Probability Distributions):**
   ```bash
   cd part1
   jupyter notebook Probability_Distributions.ipynb
   ```
   Ensure `medical_insurance_2026_kaggle.csv` is in the same directory.

2. **For Part 2 (Bayesian Probability):**
   ```bash
   cd part2
   jupyter notebook "Probability_by_bayes_theorem (1).ipynb"
   ```
   Note: Update the dataset path if running outside Google Colab.

3. **For Part 4 (Gradient Descent):**
   ```bash
   cd part4
   jupyter notebook gradientDescent_calculation.ipynb
   ```

## 📊 Requirements

```
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
scipy >= 1.7.0
jupyter >= 1.0.0
```

## 🎯 Learning Objectives

By completing this project, you will:
- ✅ Understand and implement probability distributions from mathematical formulas
- ✅ Apply Bayes' theorem to real-world classification problems
- ✅ Manually compute gradients for optimization
- ✅ Visualize mathematical concepts through code
- ✅ Bridge the gap between theoretical mathematics and practical machine learning

## 📝 Key Takeaways

1. **Part 1** demonstrates that complex probability distributions can be built from fundamental formulas
2. **Part 2** shows how Bayesian reasoning provides a principled approach to uncertainty
3. **Part 4** reveals the mechanics behind optimization algorithms used in neural networks

## 🔗 Mathematical Foundations

This project reinforces:
- Multivariate statistics and probability theory
- Conditional probability and Bayesian inference
- Calculus (derivatives and gradients)
- Optimization theory
- Statistical estimation

## 👥 Contributors

Group Work - Formative 3

## 📅 Date

March 2026