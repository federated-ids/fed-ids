# Federated Learning for IoT Intrusion Detection (CYBRIA)

A federated learning framework for intrusion detection, demonstrating privacy-preserving model training on distributed network traffic datasets using the CYBRIA IoT dataset.

## Project Overview

This project implements a federated learning-based intrusion detection system (IDS) using logistic regression. It demonstrates:

- **Centralized baseline**: Traditional logistic regression on pooled data
- **Federated learning**: Distributed training across multiple clients
- **FedAvg aggregation**: Federated averaging algorithm for model aggregation
- **Performance comparison**: Accuracy metrics for both approaches

## Project Structure

```
fed-ids/
├── data/                    # Dataset directory (place cybria.csv here)
├── federated/               # Main package
│   ├── __init__.py
│   ├── client.py           # FederatedClient class
│   ├── server.py           # FederatedServer and ModelParams classes
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── utils.py            # Utility functions (batch generator)
│   └── exceptions.py       # Custom exceptions
├── tests/                   # Unit tests
│   ├── __init__.py
│   └── test_federated.py
├── main.ipynb              # Main Jupyter notebook
├── requirements.txt        # Python dependencies
└── README.md
```

## Prerequisites

- Python 3.12 or 3.13
- Git
- Jupyter (installed via requirements.txt)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/federated-ids-cybria.git
cd federated-ids-cybria
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3.12 -m venv .venv   # or python3.13 if installed

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify Python version
python --version  # Should show 3.12 or 3.13
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download CYBRIA Dataset

1. Go to Kaggle and download the **CYBRIA Federated Learning Network Security IoT** dataset
2. Place the CSV file in the `data/` directory as `data/cybria.csv`
3. Ensure the CSV has:
   - A `Label` column with 0/1 values (normal/attack)
   - Multiple numeric feature columns

## Running the Project

### Run Tests

```bash
pytest
```

### Run the Notebook

```bash
jupyter notebook
```

Then open `main.ipynb` and run all cells. The notebook will:

1. Load and inspect the CYBRIA dataset
2. Train a centralized logistic regression baseline
3. Set up federated clients and server
4. Run federated training for 5 rounds
5. Visualize accuracy progression across rounds

## Key Components

### FederatedClient

Represents a single federated learning client with local IoT traffic data. Each client:
- Trains a local logistic regression model
- Exchanges model parameters with the server
- Evaluates performance on local data

### FederatedServer

Coordinates federated training across multiple clients:
- Aggregates client model parameters using FedAvg
- Broadcasts global model to all clients
- Manages training rounds

### ModelParams

Container for logistic regression parameters (coefficients and intercept) with support for:
- Addition (`+`) for parameter aggregation
- Division (`/`) for averaging

## Dataset Requirements

The CYBRIA dataset CSV should contain:
- **Label column**: Binary classification labels (0 = normal, 1 = attack)
- **Feature columns**: Numeric columns representing network traffic features
- The code automatically selects up to 20 numeric features (excluding the label)

If your dataset uses a different label column name, update:
- `REQUIRED_COLS` in `federated/data_loader.py`
- `label_col` parameter in function calls

## Testing

The project includes unit tests for:
- ModelParams operator overloading (`+` and `/`)
- Feature column selection (filters non-numeric and label columns)

Run tests with:
```bash
pytest
```

## Contributing

This is a collaborative project. Both team members should:
1. Make commits to the repository
2. Follow the project structure
3. Test changes before pushing

## License

See LICENSE file for details.

# Federated Learning for IoT Intrusion Detection (CYBRIA)

A Python-based federated learning framework demonstrating privacy-preserving intrusion detection on IoT network traffic using the **CYBRIA** dataset. 
This project implements a complete end-to-end FL workflow using **logistic regression**, simulating multiple distributed clients and centralized aggregation using the **FedAvg** algorithm.

---
## **Project Overview**

Modern IoT environments create massive volumes of distributed network traffic data. Centralizing this data for intrusion detection can violate privacy constraints and increase attack surface.  
Federated learning (FL) provides a privacy-preserving solution by training locally on each device and sharing **model parameters only**—not the raw data.

This project demonstrates:

- **Centralized baseline model**  
  Logistic regression trained on pooled CYBRIA data.

- **Federated Learning Simulation**  
  Sharding the dataset across clients, each with its own train/test split.

- **FedAvg aggregation**  
  Weighted averaging of client model parameters to form a global model.

- **Performance Tracking**  
  Accuracy across federated rounds, visualized in the notebook.

The final notebook walks through the entire training flow and validates correctness of the FL system.

---
## **Project Structure**

```
fed-ids/
├── data/                      # Dataset directory (cybria.csv placed here)
├── federated/
│   ├── __init__.py
│   ├── client.py              # FederatedClient class (local training + eval)
│   ├── server.py              # FederatedServer + ModelParams used in FedAvg
│   ├── data_loader.py         # CSV loading, validation, preprocessing
│   ├── utils.py               # Batch generator and helper utilities
│   └── exceptions.py          # Custom exceptions for data errors
├── tests/
│   ├── __init__.py
│   └── test_federated.py      # Pytest unit tests
├── main.ipynb                 # Primary runnable notebook
├── requirements.txt           # Dependencies
└── README.md
```

---
## **Prerequisites**

- Python **3.12** or **3.13**
- Git
- Jupyter Notebook or JupyterLab
- `pip` for dependency installation

---
## **Setup Instructions**

### **1. Clone the Repository**

```bash
git clone https://github.com/<your-username>/fed-ids.git
cd fed-ids
```

### **2. Create and Activate Virtual Environment**

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
python --version                 # Should show Python 3.12+
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Download & Add CYBRIA Dataset**

1. Download from Kaggle: **CYBRIA Federated Learning Network Security IoT**
2. Extract or convert to a single CSV file.
3. Place it as:

```
data/cybria.csv
```

### **Dataset Requirements**

Your CSV must contain:

- **attack** column (string attack names)
- Numeric traffic features

The project automatically converts `attack` → binary label:

```
0 = Normal traffic
1 = Attack traffic
```

Up to **20 numeric features** are automatically selected.

---
## **Running the Project**

### **Run Pytests**

```bash
pytest
```

Included tests validate:

- Operator overloading in `ModelParams` (`+` and `/`)
- Proper numeric feature filtering in `select_feature_columns`

---

### **Run the Notebook**

```bash
jupyter notebook
```

Then open `main.ipynb` and run all cells in order.  
The notebook will:

1. Load and validate `cybria.csv`
2. Select numeric feature columns
3. Build a centralized logistic regression baseline
4. Split dataset across multiple simulated FL clients
5. Train clients locally with a train/test split
6. Perform multiple FL rounds with FedAvg aggregation
7. Plot average accuracy per round

---
## **Key Components**

### **FederatedClient**
Each client:

- Holds a shard of the global dataset
- Performs local **train/test split**
- Trains a local logistic regression model
- Shares only model parameters (never raw data)
- Evaluates accuracy on its **local test set**

### **FederatedServer**
The server:

- Broadcasts global model parameters to clients
- Receives updated parameters from all clients
- Aggregates them using **FedAvg**
- Tracks per-round client accuracies
- Stores global model parameters after each round

### **ModelParams**
A lightweight class storing:

- Logistic regression **coefficients**
- Logistic regression **intercept**

Supports:

- **`__add__`** for parameter summation  
- **`__truediv__`** for weighted averaging  
- **`__str__`** for clean printing  

Operator overloading enables simple FedAvg math:

```python
avg = (params1 + params2 + params3) / 3
```

---
## **Accuracy Behavior (Important)**

CYBRIA’s numeric features are highly separable for binary classification.  
A centralized logistic regression already reaches **~100% test accuracy**, and the federated version converges to the same value because each client receives a random shard of the dataset.

This is expected for CYBRIA and does **not** indicate an error in your FL implementation.

A full explanation of this behavior is included in the notebook.


---
## **License**

MIT License (see `LICENSE` for details).