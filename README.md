# Federated Learning for IoT Intrusion Detection (CYBRIA)
A federated learning framework for intrusion detection, demonstrating privacy-preserving model training on distributed network traffic datasets using the CYBRIA IoT dataset.

## Team Members
- Dustin Haggett - dhaggett@stevens.edu
- Mateusz Marciniak - mmarcin1@stevens.edu


## Project Overview

This project explores a federated learning approach to intrusion detection using logistic regression. It demonstrates:
- Centralized baseline using logistic regression on pooled data
- Federated learning with multiple simulated clients
- FedAvg aggregation for combining client models
- Performance comparison between centralized and federated setups

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

## Contributions

### Dustin Haggett
- Set up the initial project structure and README
- Drafted the main modules
- Fixed bugs in the client and data loader code

### Mateusz Marciniak
- Did the initial research and came up with the project idea and requirements
- Added EDA visualizations to explore the dataset in the notebook
- Implemented the advanced aggregation methods (median, trimmed mean, weighted) in the server
- Made the training configurable so you can switch between different aggregation methods
- Fixed bugs in the weighted aggregation code
- Added examples in the notebook showing how to use the different aggregation strategies

## License

See LICENSE file for details.
