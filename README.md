# Customer Segmentation with KMeans Clustering | Machine Learning Project

**Customer Segmentation Project** — Unsupervised machine learning pipeline for segmenting customers using **KMeans clustering**, with an interactive **Streamlit** dashboard for data visualization and PCA-based cluster exploration. Built with Python, scikit-learn, and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io)

---

## Author & Contact

| | |
|---|---|
| **Author** | KuchikiRenji |
| **Email** | [KuchikiRenji@outlook.com](mailto:KuchikiRenji@outlook.com) |
| **GitHub** | [github.com/KuchikiRenji](https://github.com/KuchikiRenji) |
| **Discord** | `kuchiki_renji` |

---

## What This Project Does

This repository provides a complete **customer segmentation** workflow:

- **Data preprocessing** — Clean, encode, and scale customer data (StandardScaler, LabelEncoder).
- **KMeans clustering** — Segment customers and evaluate quality with Silhouette Score.
- **PCA visualization** — Reduce dimensions and plot clusters in 2D with Seaborn/Matplotlib.
- **Streamlit dashboard** — Interactive app to explore data, change cluster count, and view PCA plots in real time.

Useful for **marketing analytics**, **customer behavior analysis**, and **unsupervised learning** practice.

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation & Run

```bash
git clone https://github.com/KuchikiRenji/customer-segmentation-project.git
cd customer-segmentation-project
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`) to use the dashboard.

---

## Project Structure

```
customer-segmentation-project/
├── data/
│   ├── customers.csv                # Raw customer dataset
│   ├── customers_preprocessed.csv    # Preprocessed dataset
├── src/
│   ├── data_preprocessing.py        # Cleaning, encoding, scaling
│   ├── clustering.py                # KMeans clustering & Silhouette evaluation
│   ├── utils.py                     # Data load/save utilities
│   ├── visualization.py             # PCA and cluster plots
├── app/
│   ├── dashboard.py                 # Streamlit interactive dashboard
├── case study/
│   ├── R&D(ML)_DeeptiGupta.pdf      # Case study documentation
├── images/
│   ├── preprocessing.png            # Preprocessing visualization
│   ├── clustering.png               # Clustering output example
│   ├── dashboard.png                # Dashboard screenshot
├── requirements.txt
└── README.md
```

---

## Project Workflow

### 1. Data Preprocessing

`data_preprocessing.py` handles missing values, encodes categorical variables, and scales numerical data with `StandardScaler`. Output is saved under `data/`.

**Example:**

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')
```

![Preprocessing](https://github.com/KuchikiRenji/customer-segmentation-project/blob/main/images/preprocessing.png)

### 2. KMeans Clustering

`clustering.py` runs KMeans and evaluates clusters using the Silhouette Score.

**Example:**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_kmeans_clustering(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(df)
    return labels, kmeans
```

![Clustering](https://github.com/KuchikiRenji/customer-segmentation-project/blob/main/images/clustering.png)

### 3. Interactive Dashboard

`app/dashboard.py` is a Streamlit app that lets you:

- View sample data
- Change the number of clusters
- Visualize clusters in a 2D PCA plot

```bash
streamlit run app/dashboard.py
```

![Dashboard](https://github.com/KuchikiRenji/customer-segmentation-project/blob/main/images/dashboard.png)
![Dashboard1](https://github.com/KuchikiRenji/customer-segmentation-project/blob/main/images/dashboard1.png)

### 4. Cluster Visualization

`visualization.py` uses PCA for dimensionality reduction and Seaborn for scatter plots.

**Example:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df, labels):
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette="viridis")
    plt.show()
```

### 5. Utility Functions

`utils.py` provides:

- `save_data` — Save DataFrame to CSV
- `load_processed_data` — Load preprocessed data

---

## Case Study

The `case study/` folder contains **R&D(ML)_DeeptiGupta.pdf** with:

- Problem statement
- Task 1 questions and answers

---

## Key Results & Metrics

- **Silhouette Score** — Cluster cohesion and separation
- **PCA plots** — 2D interpretation of segments
- **Interactive dashboard** — Real-time cluster exploration

---

## Technologies Used

- **Python** — Core language
- **pandas** — Data handling
- **NumPy** — Numerical operations
- **scikit-learn** — KMeans, PCA, StandardScaler, Silhouette Score
- **Streamlit** — Web dashboard
- **Matplotlib & Seaborn** — Visualizations

---

## Contributing

Contributions are welcome. Open an issue or submit a pull request on [GitHub](https://github.com/KuchikiRenji/customer-segmentation-project).

---

## License

See [LICENSE](LICENSE) in this repository.

---

*Customer Segmentation Project by KuchikiRenji — [GitHub](https://github.com/KuchikiRenji) | [Email](mailto:KuchikiRenji@outlook.com) | Discord: kuchiki_renji*
