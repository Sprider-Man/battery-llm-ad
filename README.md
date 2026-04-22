We have made the dataset public and it can be obtained from this link. 
https://pan.baidu.com/s/1G0MJKrqyFRdXbpMLbTtgNg code: wrux

# Data Analysis Toolkit

This repository contains two core modules for analyzing vehicle battery and telemetry data: **Rule Mining** and **Anomaly Detection**. It is designed to help data scientists and engineers detect anomalies and extract meaningful patterns from large datasets.

## Modules

### 1. Rule Mining

The Rule Mining module extracts frequent patterns and rules from structured datasets. It is useful for understanding associations between features and discovering hidden insights in telemetry data.

**Key Features:**
- Frequent pattern extraction
- Association rule generation
- Configurable thresholds for support and confidence
- Integration with Pandas and NumPy for flexible data handling

**Usage Example:**
```python
from rule_mining import RuleMiner

# Initialize the miner with dataset and parameters
miner = RuleMiner(
    data_path="data/vehicle_data.xlsx",
    min_support=0.05,
    min_confidence=0.7
)

# Mine rules from the dataset
rules = miner.mine_rules()
print(rules)
```

**Functions:**
- `load_existing_anomaly_data(data_path)` — Load dataset and preprocess it by removing dependent columns while retaining essential columns such as vehicle state, battery voltage, and anomaly flags.
- Preprocessing steps include handling Excel (`.xlsx`, `.xls`) and JSON files.

### 2. Anomaly Detection

The Anomaly Detection module identifies unusual or unexpected behavior in vehicle telemetry data, including battery voltage jumps, current spikes, and temperature anomalies.

**Key Features:**
- Detect anomalies in voltage, current, and temperature
- Supports different detection methods (threshold-based, statistical)
- Provides visualization of detected anomalies
- Configurable sensitivity and thresholds

**Usage Example:**
```python
from anomaly_detection import AnomalyDetector

# Initialize the detector
detector = AnomalyDetector(
    data_path="data/vehicle_data.xlsx",
    method="threshold"  # threshold-based anomaly detection
)

# Detect anomalies
anomalies = detector.detect()
print(anomalies)
```

**Thresholds (from code defaults):**
- `MONO_VOLTAGE_JUMP_THRESHOLD = 0.05` V  
- `CURRENT_JUMP_THRESHOLD = 0.5` A  
- `TEMPERATURE_JUMP_THRESHOLD = 3.0` °C  
- `CHARGE_TOTAL_VOLTAGE_JUMP = 3.0` V  

**Functions:**
- Load Excel or JSON datasets
- Preprocess key telemetry columns
- Detect anomalies based on voltage, current, and temperature jumps

## Installation

1. Clone this repository:
```bash
git clone https://github.com/d2236355239-ux/battery-llm-ad.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- Python 3.8+
- Pandas
- NumPy
- Matplotlib
- OpenPyXL (for `.xlsx` support)
- Swift (for rule mining and model utilities)

## Configuration

- GPU configuration is set via `CUDA_VISIBLE_DEVICES` environment variable.  
- Logger is configured via `get_logger()` for debugging and tracking.

## License

This project is licensed under the MIT License.
