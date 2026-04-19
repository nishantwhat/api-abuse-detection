"""
data_loader.py
--------------
Responsible for loading or generating the dataset used for API abuse detection.

Strategy: Hybrid approach
- Simulates the structure of UNSW-NB15 / CICIDS network intrusion datasets
- Adds API-specific context through feature engineering in the next module
- Generates realistic synthetic data that mirrors network + API telemetry

In a real deployment, you would replace generate_hybrid_dataset() with a
function that loads actual UNSW-NB15 or CICIDS CSV exports from the /data folder.
"""

import numpy as np
import pandas as pd
import os


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file (e.g., exported UNSW-NB15 or CICIDS data).
    Falls back to synthetic generation if file is not found.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A pandas DataFrame with the raw data.
    """
    if os.path.exists(filepath):
        print(f"[INFO] Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded {len(df)} records with {df.shape[1]} columns.")
        return df
    else:
        print(f"[WARNING] File not found: {filepath}")
        print("[INFO] Falling back to synthetic hybrid dataset generation...")
        return generate_hybrid_dataset()


def generate_hybrid_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors network intrusion datasets
    adapted for API traffic monitoring.

    Class distribution (mimicking real-world API traffic imbalance):
        0 = Normal    → 60% of traffic
        1 = Suspicious → 25% of traffic
        2 = Abuse      → 15% of traffic

    Each class has distinct statistical profiles to make classification
    meaningful (not random). This mirrors real patterns like:
        - Normal: low rate, varied endpoints, low failed auth
        - Suspicious: moderate rate spikes, some failed auth
        - Abuse: high rate, concentrated endpoints, high failed auth

    Args:
        n_samples: Total number of records to generate.
        random_state: Seed for reproducibility.

    Returns:
        A pandas DataFrame representing raw API traffic features.
    """
    rng = np.random.default_rng(random_state)

    # Define how many samples per class
    n_normal     = int(n_samples * 0.60)
    n_suspicious = int(n_samples * 0.25)
    n_abuse      = n_samples - n_normal - n_suspicious

    def make_class(n, profile):
        """
        Generate n rows for one class using the given statistical profile.
        Each feature is drawn from a distribution tuned to that class.
        """
        return {
            # How many requests per minute from this IP/session
            "request_rate": rng.normal(profile["req_rate_mean"], profile["req_rate_std"], n).clip(0.1, 500),

            # How spread out are the endpoint hits (high = scattered, low = focused attack)
            "endpoint_hit_variance": rng.normal(profile["ep_var_mean"], profile["ep_var_std"], n).clip(0, 1),

            # Ratio of failed authentication attempts to total auth attempts
            "failed_auth_ratio": rng.beta(profile["auth_a"], profile["auth_b"], n),

            # Average byte size of the request payload
            "avg_payload_size": rng.normal(profile["payload_mean"], profile["payload_std"], n).clip(50, 10000),

            # Seconds between consecutive requests (lower = more aggressive)
            "time_between_requests": rng.exponential(profile["tbr_scale"], n).clip(0.05, 300),

            # How many distinct IPs are involved (proxy/botnet signal)
            "ip_request_diversity": rng.normal(profile["ip_div_mean"], profile["ip_div_std"], n).clip(1, 200).astype(int),

            # Entropy of User-Agent strings seen (high = rotating agents = bot signal)
            "user_agent_entropy": rng.normal(profile["ua_entropy_mean"], profile["ua_entropy_std"], n).clip(0, 1),

            # HTTP verb distribution (POST ratio — abuse often has high POST for injections)
            "http_verb_post_ratio": rng.beta(profile["verb_a"], profile["verb_b"], n),

            # Proportion of 4xx responses (auth failures, not found, forbidden)
            "error_4xx_ratio": rng.beta(profile["err4_a"], profile["err4_b"], n),

            # Proportion of 5xx responses (server errors caused by injection attempts)
            "error_5xx_ratio": rng.beta(profile["err5_a"], profile["err5_b"], n),

            # Payload size standard deviation (consistent = bot, variable = human)
            "payload_size_std": rng.exponential(profile["payload_std_scale"], n).clip(0, 3000),

            # Session duration in seconds
            "session_duration": rng.normal(profile["session_mean"], profile["session_std"], n).clip(1, 7200),
        }

    # Statistical profiles per class — tuned to simulate realistic API behavior
    normal_profile = dict(
        req_rate_mean=12, req_rate_std=5,
        ep_var_mean=0.75, ep_var_std=0.15,
        auth_a=1, auth_b=20,
        payload_mean=512, payload_std=200,
        tbr_scale=8,
        ip_div_mean=2, ip_div_std=1,
        ua_entropy_mean=0.15, ua_entropy_std=0.1,
        verb_a=2, verb_b=5,
        err4_a=1, err4_b=15,
        err5_a=1, err5_b=50,
        payload_std_scale=300,
        session_mean=400, session_std=150,
    )

    suspicious_profile = dict(
        req_rate_mean=60, req_rate_std=20,
        ep_var_mean=0.45, ep_var_std=0.2,
        auth_a=3, auth_b=10,
        payload_mean=800, payload_std=400,
        tbr_scale=2,
        ip_div_mean=15, ip_div_std=8,
        ua_entropy_mean=0.55, ua_entropy_std=0.2,
        verb_a=3, verb_b=4,
        err4_a=4, err4_b=10,
        err5_a=2, err5_b=20,
        payload_std_scale=700,
        session_mean=180, session_std=80,
    )

    abuse_profile = dict(
        req_rate_mean=200, req_rate_std=50,
        ep_var_mean=0.1, ep_var_std=0.08,
        auth_a=8, auth_b=3,
        payload_mean=2500, payload_std=800,
        tbr_scale=0.3,
        ip_div_mean=80, ip_div_std=30,
        ua_entropy_mean=0.9, ua_entropy_std=0.08,
        verb_a=6, verb_b=2,
        err4_a=7, err4_b=4,
        err5_a=5, err5_b=5,
        payload_std_scale=1500,
        session_mean=60, session_std=30,
    )

    # Build each class subset
    records_normal     = make_class(n_normal, normal_profile)
    records_suspicious = make_class(n_suspicious, suspicious_profile)
    records_abuse      = make_class(n_abuse, abuse_profile)

    # Combine into DataFrames and assign labels
    df_normal     = pd.DataFrame(records_normal);     df_normal["label"]     = 0
    df_suspicious = pd.DataFrame(records_suspicious); df_suspicious["label"] = 1
    df_abuse      = pd.DataFrame(records_abuse);      df_abuse["label"]      = 2

    df = pd.concat([df_normal, df_suspicious, df_abuse], ignore_index=True)

    # Shuffle so classes aren't in order (avoids order bias during train/test split)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Introduce ~3% missing values to simulate real-world data quality issues
    _inject_missing_values(df, missing_fraction=0.03, rng=rng)

    print(f"[INFO] Generated hybrid dataset: {len(df)} records, {df.shape[1]} columns.")
    print(f"[INFO] Class distribution:\n{df['label'].value_counts().sort_index().to_string()}")

    return df


def _inject_missing_values(df: pd.DataFrame, missing_fraction: float, rng) -> None:
    """
    Randomly set some cells to NaN to simulate data collection gaps.
    Only applied to feature columns, not the label.

    Args:
        df: The DataFrame to modify in place.
        missing_fraction: Fraction of total cells to set as missing.
        rng: NumPy random generator.
    """
    feature_cols = [c for c in df.columns if c != "label"]
    total_cells  = len(df) * len(feature_cols)
    n_missing    = int(total_cells * missing_fraction)

    for _ in range(n_missing):
        row = rng.integers(0, len(df))
        col = rng.choice(feature_cols)
        df.at[row, col] = np.nan
