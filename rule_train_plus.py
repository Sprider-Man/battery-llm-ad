import os
import pandas as pd
import numpy as np
import re
from swift.utils import get_logger
import matplotlib.pyplot as plt

# Global Configuration (English Version)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = get_logger()
# Use English font, no Chinese support needed
plt.rcParams["font.family"] = ["Arial", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # Support negative sign display
# Create output directory
os.makedirs('F:\\data', exist_ok=True)


def load_existing_anomaly_data(data_path):
    """Load data (English Version)"""
    try:
        file_ext = os.path.splitext(data_path)[1].lower()
        if file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path, engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
        elif file_ext == '.json':
            df = pd.read_json(data_path, orient='records')
        else:
            logger.error(f"Unsupported format: {file_ext}, only .xlsx/.json are supported")
            return None, None

        # Core columns (Original Chinese column names, keep unchanged for data matching)
        required_cols = ['整车State状态（状态机编码）', '动力电池内部总电压V1', 'is_anomaly',
                         '动力电池充/放电电流', '1号温度检测点温度']

        # Auto detect cell voltage columns
        mono_voltage_cols = [col for col in df.columns if '号电池单体电压' in col]
        if not mono_voltage_cols:
            logger.error("No cell voltage columns found (column name must contain '号电池单体电压')")
            return None, None

        logger.info(f"Found cell voltage columns: {mono_voltage_cols}, total: {len(mono_voltage_cols)}")

        # Check missing columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing core columns: {missing_cols}")
            return None, None

        # Filter valid data
        df = df[
            (df['is_anomaly'].notna()) &
            (df['动力电池内部总电压V1'].notna()) &
            (df['整车State状态（状态机编码）'].notna()) &
            (df['动力电池充/放电电流'].notna()) &
            (df['1号温度检测点温度'].notna())
            ].reset_index(drop=True)

        df['整车State状态（状态机编码）'] = df['整车State状态（状态机编码）'].astype(int)
        df['is_anomaly'] = df['is_anomaly'].astype(bool)

        logger.info(f"Data loaded successfully: {len(df)} records, {df['is_anomaly'].sum()} abnormal samples")
        return df, mono_voltage_cols
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def calculate_diff_distribution(df, col_name, is_mono=False):
    """Calculate adjacent difference distribution"""
    df_diff = df.copy()
    if is_mono:
        all_diffs = []
        for col in col_name:
            diffs = df_diff[col].diff().abs().dropna()
            all_diffs.extend(diffs.tolist())
        all_diffs = pd.Series(all_diffs, name='mono_diff')
    else:
        all_diffs = df_diff[col_name].diff().abs().dropna()
    return all_diffs


def determine_threshold(normal_diffs, abnormal_diffs, confidence=0.99):
    """Determine threshold based on normal sample distribution"""
    if len(normal_diffs) == 0:
        logger.warning("Normal sample differences are empty, use default threshold")
        return 0.1

    normal_diffs = normal_diffs.dropna()
    abnormal_diffs = abnormal_diffs.dropna()

    if len(normal_diffs) == 0:
        logger.warning("Normal sample differences empty after filtering, use default threshold")
        return 0.1

    threshold = normal_diffs.quantile(confidence)

    if pd.isna(threshold):
        threshold = normal_diffs.max()
        logger.info(f"Quantile is NaN, use max value of normal samples: {threshold}")

    if len(abnormal_diffs) > 0:
        abnormal_ratio = (abnormal_diffs > threshold).mean()
        if abnormal_ratio < 0.3:
            threshold = np.percentile(np.concatenate([normal_diffs.values, abnormal_diffs.values]), 95)
            logger.info(f"Abnormal ratio too low ({abnormal_ratio:.2%}), adjust threshold to 95th percentile: {threshold}")
    else:
        logger.warning("No abnormal sample differences, calculate threshold based on normal samples only")

    return round(float(threshold), 3)


def extract_standard_thresholds(df, mono_voltage_cols):
    """Extract anomaly detection thresholds from data distribution"""
    normal_df = df[df['is_anomaly'] == False].copy()
    abnormal_df = df[df['is_anomaly'] == True].copy()

    logger.info(f"Normal samples: {len(normal_df)}, Abnormal samples: {len(abnormal_df)}")

    if len(normal_df) == 0 or len(abnormal_df) == 0:
        logger.error("Normal/abnormal sample count is 0, cannot calculate thresholds")
        return None

    # Charge/Discharge status codes
    discharge_states = [30]
    charge_states = [20]
    discharge_df = df[df['整车State状态（状态机编码）'].isin(discharge_states)]
    charge_df = df[df['整车State状态（状态机编码）'].isin(charge_states)]

    logger.info(f"Charge samples: {len(charge_df)}, Discharge samples: {len(discharge_df)}")

    thresholds = {}

    # 1. Cell Voltage Jump Threshold
    normal_mono_diffs = calculate_diff_distribution(normal_df, mono_voltage_cols, is_mono=True)
    abnormal_mono_diffs = calculate_diff_distribution(abnormal_df, mono_voltage_cols, is_mono=True)
    thresholds['mono_voltage'] = determine_threshold(normal_mono_diffs, abnormal_mono_diffs) if len(normal_mono_diffs) > 0 else 0.05

    # 2. Temperature Jump Threshold
    normal_temp_diffs = calculate_diff_distribution(normal_df, '1号温度检测点温度')
    abnormal_temp_diffs = calculate_diff_distribution(abnormal_df, '1号温度检测点温度')
    thresholds['temperature'] = determine_threshold(normal_temp_diffs, abnormal_temp_diffs) if len(normal_temp_diffs) > 0 else 3.0

    # 3. Charge Current Jump Threshold
    if len(charge_df) > 0:
        normal_curr_diffs = calculate_diff_distribution(charge_df[charge_df['is_anomaly'] == False], '动力电池充/放电电流')
        abnormal_curr_diffs = calculate_diff_distribution(charge_df[charge_df['is_anomaly'] == True], '动力电池充/放电电流')
        thresholds['current'] = determine_threshold(normal_curr_diffs, abnormal_curr_diffs) if len(normal_curr_diffs) > 0 else 0.5
    else:
        thresholds['current'] = 0.5

    # 4. Discharge Total Voltage Overlimit Threshold
    if len(discharge_df) > 0:
        normal_dis_volt = discharge_df[discharge_df['is_anomaly'] == False]['动力电池内部总电压V1']
        thresholds['discharge_total'] = round(float(normal_dis_volt.quantile(0.99)), 1) if len(normal_dis_volt) > 0 else 378.2
    else:
        thresholds['discharge_total'] = 378.2

    # Store numeric values for plotting + formatted values for rules
    thresholds_numeric = thresholds.copy()
    thresholds['mono_voltage'] = f"{thresholds['mono_voltage']:.2f}V"
    thresholds['temperature'] = f"{thresholds['temperature']:.1f}℃"
    thresholds['current'] = f"{thresholds['current']:.1f}A"
    thresholds['discharge_total'] = f"{thresholds['discharge_total']:.1f}V"

    return thresholds, thresholds_numeric


def generate_exact_rule_text(thresholds):
    """Generate English anomaly detection rules"""
    return f"""# Battery Data Anomaly Detection Rules

### 1. Common Anomaly Rules (Charge & Discharge)

| Anomaly Type               | Threshold       | Rule Description |
|----------------------------|-----------------|------------------|
| Cell Voltage Jump          | {thresholds['mono_voltage']} | Calculate voltage difference between consecutive rows; mark as abnormal if absolute value exceeds threshold |
| Temperature Jump           | {thresholds['temperature']} | Calculate temperature difference between consecutive rows; mark as abnormal if absolute value exceeds threshold |

### 2. Charge-Only Anomaly Rules

| Anomaly Type               | Threshold       | Rule Description |
|----------------------------|-----------------|------------------|
| Charging Current Jump      | {thresholds['current']} | Calculate current difference between consecutive rows; mark as abnormal if absolute value exceeds threshold |

### 3. Discharge-Only Anomaly Rules

| Anomaly Type               | Threshold       | Rule Description |
|----------------------------|-----------------|------------------|
| Total Discharge Voltage Overlimit | {thresholds['discharge_total']} | Mark as abnormal if total battery voltage exceeds threshold |"""


def plot_diff_distribution(normal_diffs, abnormal_diffs, title, xlabel, save_path):
    """English version: Plot difference distribution comparison"""
    plt.figure(figsize=(10, 6))
    plt.hist(normal_diffs, bins=50, alpha=0.5, label='Normal Samples', density=True, color='#1f77b4')
    plt.hist(abnormal_diffs, bins=50, alpha=0.5, label='Abnormal Samples', density=True, color='#ff4b5c')
    if len(normal_diffs) > 0:
        plt.axvline(normal_diffs.quantile(0.99), color='red', linestyle='--', linewidth=2, label='Threshold (99th Percentile)')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Distribution chart saved to: {save_path}")


def plot_threshold_line_chart(thresholds_numeric, save_path):
    """
    Core function: Plot ENGLISH threshold line chart
    X-axis: Threshold Type
    Y-axis: Default Fallback Threshold Value
    """
    # Define threshold labels (English) and corresponding values
    threshold_types = [
        'Cell Voltage Jump',
        'Temperature Jump',
        'Charging Current Jump',
        'Total Discharge Voltage Overlimit'
    ]
    threshold_values = [
        thresholds_numeric['mono_voltage'],
        thresholds_numeric['temperature'],
        thresholds_numeric['current'],
        thresholds_numeric['discharge_total']
    ]

    # Create professional line chart
    plt.figure(figsize=(12, 7))
    plt.plot(threshold_types, threshold_values, marker='o', linewidth=3, markersize=8,
             color='#0066cc', markerfacecolor='#00ccff', markeredgecolor='#003366', markeredgewidth=2)

    # Add value labels on data points
    for i, value in enumerate(threshold_values):
        plt.annotate(f'{value}', (i, value), textcoords="offset points", xytext=(0,10),
                     ha='center', fontsize=11, fontweight='bold', color='#003366')

    # Chart style (English)
    plt.title('Default Fallback Threshold by Threshold Type', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Threshold Type', fontsize=13, fontweight='bold')
    plt.ylabel('Default Fallback Threshold Value', fontsize=13, fontweight='bold')
    plt.xticks(rotation=15, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save high-resolution image
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"English threshold line chart saved to: {save_path}")


def main():
    # 1. Load data
    DATA_PATH = r"F:\data\new_data_anomalies_debug.xlsx"
    df, mono_voltage_cols = load_existing_anomaly_data(DATA_PATH)
    if df is None:
        logger.error("Data loading failed")
        exit(1)

    # 2. Calculate thresholds
    thresholds, thresholds_numeric = extract_standard_thresholds(df, mono_voltage_cols)
    if thresholds is None:
        logger.error("Threshold calculation failed")
        exit(1)

    # 3. Generate English rules
    rule_text = generate_exact_rule_text(thresholds)
    rule_save_path = os.path.join('F:\\data', 'battery_anomaly_rules_english.txt')
    with open(rule_save_path, 'w', encoding='utf-8') as f:
        f.write(rule_text)

    # 4. Plot charts (ENGLISH VERSION)
    # 4.1 Cell voltage difference distribution
    normal_df = df[df['is_anomaly'] == False]
    abnormal_df = df[df['is_anomaly'] == True]
    normal_mono_diffs = calculate_diff_distribution(normal_df, mono_voltage_cols, is_mono=True)
    abnormal_mono_diffs = calculate_diff_distribution(abnormal_df, mono_voltage_cols, is_mono=True)
    plot_diff_distribution(
        normal_mono_diffs, abnormal_mono_diffs,
        title='Cell Voltage Jump Difference Distribution (Normal vs Abnormal)',
        xlabel='Voltage Difference (V)',
        save_path=os.path.join('F:\\data', 'cell_voltage_distribution_en.png')
    )

    # 4.2 Core: English threshold line chart (WHAT YOU NEED)
    plot_threshold_line_chart(
        thresholds_numeric,
        save_path=os.path.join('F:\\data', 'threshold_line_chart_english.png')
    )

    # 5. Print results
    print("=" * 90)
    print("Auto-generated Anomaly Detection Thresholds (English)")
    print("=" * 90)
    print(rule_text)
    logger.info(f"English rules saved to: {rule_save_path}")
    logger.info("All English charts generated successfully!")


if __name__ == "__main__":
    main()