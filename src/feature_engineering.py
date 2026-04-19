"""
feature_engineering.py
-----------------------
Transforms raw API telemetry features into higher-level signals that
classification models can more effectively learn from.

Grounded in real-world practices described in:
  - OWASP API Security Top 10 (BOLA, Broken Authentication)
  - Cloudflare bot detection (TLS fingerprinting, behavioral entropy)
  - Research: "Feature Engineering for Malware Classification Based on
    API Call Sequences" (arXiv, 2024)

Design Philosophy:
  - Each derived feature has a clear, explainable rationale
  - No magic numbers without comments explaining the threshold
  - Features are kept interpretable (no black-box transformations)
  - Scaling is done OUTSIDE this module (in model_training.py)
    so raw engineered features remain visible for EDA
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    Applies all transformations and returns an enriched DataFrame.

    Args:
        df: Cleaned DataFrame from preprocessing.py

    Returns:
        DataFrame with original + engineered features.
    """
    print("\n[FEATURE ENGINEERING] Deriving behavioral signals...")
    df = df.copy()

    # --- Behavioral Intensity Score ---
    # Combines request rate and time-between-requests into one aggression score.
    # A high score means: many requests AND short gaps → likely automated abuse.
    # Formula: rate * (1 / tbr) normalized to avoid scale dominance
    df["aggression_score"] = (
        df["request_rate"] / (df["time_between_requests"] + 0.1)
    ).round(4)
    print("   + aggression_score       (request rate / inter-arrival time)")

    # --- Authentication Stress Index ---
    # High failed_auth_ratio + high request_rate = likely credential stuffing.
    # This mirrors the pattern OWASP identifies as Broken Authentication (API2).
    df["auth_stress_index"] = (
        df["failed_auth_ratio"] * df["request_rate"]
    ).round(4)
    print("   + auth_stress_index      (failed_auth_ratio × request_rate)")

    # --- Endpoint Focus Score ---
    # Low endpoint_hit_variance = attacker is hammering one specific endpoint.
    # Inverted so that higher score = more focused = more suspicious.
    # BOLA attacks (OWASP API1) show this pattern: /api/users/1001 repeatedly.
    df["endpoint_focus_score"] = (1 - df["endpoint_hit_variance"]).round(4)
    print("   + endpoint_focus_score   (1 - endpoint_hit_variance → focus proxy)")

    # --- IP Diversity Normalized ---
    # Scales ip_request_diversity to [0,1] range using log normalization.
    # Botnets distribute across hundreds of IPs; log dampens extreme values.
    # Max 200 IPs assumed based on dataset generation bounds.
    df["ip_diversity_norm"] = (
        np.log1p(df["ip_request_diversity"]) / np.log1p(200)
    ).round(4)
    print("   + ip_diversity_norm      (log-normalized IP spread → botnet signal)")

    # --- Payload Anomaly Score ---
    # Abnormally large OR highly variable payloads suggest injection attempts.
    # (SQL Injection, XSS, RCE payloads are often unusually large or inconsistent)
    # Normalize each component to [0,1] before combining.
    max_payload = df["avg_payload_size"].max()
    max_std     = df["payload_size_std"].max()
    df["payload_anomaly_score"] = (
        (df["avg_payload_size"] / (max_payload + 1)) * 0.5 +
        (df["payload_size_std"]  / (max_std + 1))    * 0.5
    ).round(4)
    print("   + payload_anomaly_score  (large + variable payload → injection signal)")

    # --- Error Pressure Index ---
    # High 4xx errors = repeated unauthorized access or probing.
    # High 5xx errors = server-side stress from injection/resource exhaustion.
    # Weighted: 5xx slightly more severe than 4xx in abuse scenarios.
    df["error_pressure_index"] = (
        df["error_4xx_ratio"] * 0.4 + df["error_5xx_ratio"] * 0.6
    ).round(4)
    print("   + error_pressure_index   (4xx + 5xx ratio weighted combination)")

    # --- Bot Likelihood Score ---
    # Combines entropy-based signals (user agent rotation, IP spread, high POST ratio)
    # into a composite bot probability score.
    # Rationale from Cloudflare JA4 docs: rotating UAs + POST dominance = automation.
    df["bot_likelihood_score"] = (
        df["user_agent_entropy"]   * 0.4 +
        df["ip_diversity_norm"]    * 0.3 +
        df["http_verb_post_ratio"] * 0.3
    ).round(4)
    print("   + bot_likelihood_score   (UA entropy + IP spread + POST ratio)")

    # --- Session Efficiency ---
    # Normal users have longer sessions with moderate request rates.
    # Abusers have short sessions with extreme request bursts.
    # Low value = short session, high requests = suspicious efficiency.
    df["session_efficiency"] = (
        df["request_rate"] / (df["session_duration"] + 1)
    ).round(4)
    print("   + session_efficiency     (requests per second of session)")

    feature_count = df.shape[1] - 1  # Exclude label
    print(f"\n[FEATURE ENGINEERING] Done. Total features available: {feature_count}")

    return df


def get_feature_names(df: pd.DataFrame) -> list:
    """
    Return list of all feature column names (excludes 'label').

    Args:
        df: Engineered DataFrame.

    Returns:
        List of feature column names.
    """
    return [c for c in df.columns if c != "label"]


def get_engineered_feature_names() -> list:
    """
    Return only the names of derived (engineered) features.
    Useful for targeted analysis and documentation.
    """
    return [
        "aggression_score",
        "auth_stress_index",
        "endpoint_focus_score",
        "ip_diversity_norm",
        "payload_anomaly_score",
        "error_pressure_index",
        "bot_likelihood_score",
        "session_efficiency",
    ]
