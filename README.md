# Decision Trees and Dimensionality Reduction (PCA)

A comprehensive implementation of Decision Tree classification and Principal Component Analysis (PCA) for dimensionality reduction using the Iris and Wine datasets.

## Overview

This project demonstrates fundamental machine learning concepts including:
- Decision Tree classification with different splitting criteria
- Manual calculation of Gini impurity and information gain
- K-fold cross-validation for model evaluation
- Principal Component Analysis for dimensionality reduction
- Comparative analysis of model performance

## Features

### I. Decision Tree Classification
- **Iris Dataset Analysis**: Build decision trees using petal features
- **Gini Impurity**: Manual calculation and threshold optimization
- **Entropy vs Gini**: Comparison of splitting criteria
- **Cross-Validation**: K-fold validation with k=3, 5, and 10
- **Tree Visualization**: Visual representation of decision boundaries
- **Export Capability**: Save trees in GraphViz DOT format

### II. Principal Component Analysis (PCA)
- **Iris Dataset**: Reduce 4 features to 2 principal components
- **Wine Dataset**: Adaptive reduction maintaining 95% variance
- **Explained Variance**: Visualization of cumulative variance
- **Performance Comparison**: Model accuracy before and after PCA

### III. Model Comparison
- Logistic Regression (baseline)
- Decision Tree Classifier
- Performance metrics: Accuracy, Precision, Recall, F1-Score

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mr-taha-saqib/Decision-Trees-and-Dimensionality-Reduction-PCA-.git
cd Decision-Trees-and-Dimensionality-Reduction-PCA-
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook "Decision Trees and Dimensionality Reduction (PCA).ipynb"
```

Or execute as a Python script:
```python
python decision_tree_pca.py
```

## Project Structure

```
.
├── Decision Trees and Dimensionality Reduction (PCA).ipynb
├── README.md
└── outputs/
    ├── Iris_DTree.dot
    └── LAB5_summary.md
```

## Key Concepts Covered

### Decision Trees
- **Gini Impurity**: Measure of node impurity for classification
- **Information Gain**: Reduction in entropy after a split
- **Max Depth**: Controlling tree complexity to prevent overfitting
- **Feature Importance**: Identifying most discriminative features

### Dimensionality Reduction
- **PCA**: Linear transformation to uncorrelated components
- **Variance Retention**: Balancing dimensionality vs information loss
- **Feature Scaling**: StandardScaler preprocessing for PCA
- **Visualization**: 2D projection of high-dimensional data

## Results

### Iris Dataset (Decision Tree)
- **Petal Features Only**: High accuracy with just 2 features
- **5-Fold CV**: Robust performance validation
- **Max Depth=2**: Simple, interpretable tree structure

### PCA Performance
- **Iris**: 2 components capture most variance
- **Wine**: Reduced to fewer components while maintaining 95% variance
- **Model Impact**: Minimal accuracy loss with significant dimensionality reduction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Taha Saqib

## Acknowledgments

- Scikit-learn documentation and examples
- UC Irvine Machine Learning Repository for datasets
- Machine Learning community for best practices
