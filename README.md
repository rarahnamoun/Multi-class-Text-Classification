
# Modified Naive Bayes Classifier for Imbalanced Text Classification 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen.svg)](./docs)

## 📚 Overview

This project presents a novel approach to handling imbalanced datasets in text classification tasks by introducing a modified version of the Naive Bayes classifier. Our method incorporates class probability adjustments and feature probability modifications to better handle class imbalance issues commonly found in real-world datasets.

<img src="/api/placeholder/800/400" alt="Project Architecture Overview" />

## 🔍 Key Features

- Custom Bayes Classifier with class imbalance handling
- Advanced probability adjustment mechanisms
- Dynamic probability evolution using differential equations
- Comprehensive comparison with traditional approaches
- Support for multi-class text classification

## 🧮 Mathematical Foundation

### Class Probability Adjustment

We introduce a weighted prior approach to adjust class probabilities:

```
P(y = Cₖ) = (Nₖ + α) / (N + α·K)
```

Where:
- Nₖ: Number of training instances in class Cₖ
- N: Total number of training instances
- K: Total number of classes
- α: Smoothing parameter

### Feature Probability Adjustment

Features are adjusted using modified Laplace smoothing:

```
P(xᵢ|y = Cₖ) = (Nᵢₖ + α) / (Nₖ + α·V)
```

Where:
- Nᵢₖ: Instances where feature xᵢ occurs in class Cₖ
- V: Vocabulary size
- α: Laplace smoothing parameter

## 📊 Results

my modified approach showed improvements in handling imbalanced datasets:

- 1% improvement over traditional Naive Bayes
- Better handling of minority classes
- More stable probability estimates



## 🗂️ Project Structure

```
project/
│
├── src/
│   ├
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── docs/
│   ├── report_part0.pdf
│   └── report_part1.pdf
│
├── README.md
└
```

## 📝 Implementation Details

### Custom Bayes Classifier

The implementation includes:

1. **Probability Initialization**
   - Dynamic prior probability calculation
   - Feature probability smoothing

2. **Differential Equation Integration**
   ```python
   dP(y = Cₖ)/dt = β(Nₖ(t) + α)/(N(t) + α·K) - P(y = Cₖ)
   ```

3. **Softmax Adjustment**
   ```python
   P'(xᵢ|Cₖ) = exp(α·P(xᵢ|y=Cₖ)) / Σₗ exp(α·P(xᵢ|y=Cₗ))
   ```

## 📊 Comparative Analysis

We compared our approach with:

- Traditional Naive Bayes
- LSTM-based classifiers
- BERT distillation methods



## 🔧 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/rarahnamoun/Multi-class-Text-Classification.git

# Run the notebooks
jupyter notebook src
```

## 📖 Documentation

- [Full Technical Report](./docs/report_part0.pdf)
- [Methodology Documentation](./docs/report_part1.pdf)
