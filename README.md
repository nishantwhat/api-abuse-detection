# API Abuse Detection using Machine Learning

## 1. Project Overview
This project implements a production-ready Machine Learning pipeline to detect and classify Application Programming Interface (API) abuse. By analyzing raw, nested JSON HTTP traffic logs, the system leverages supervised machine learning to identify malicious intent hidden within legitimate Layer 7 (Application) traffic. 

Unlike standard anomaly detection that only flags "bad" traffic, this system categorizes the exact nature of the threat, providing actionable intelligence for security operations.

## 2. Problem Statement
APIs form the underlying infrastructure of modern web and mobile applications. Traditional Network Intrusion Detection Systems (NIDS) and rate-limiting Web Application Firewalls (WAFs) operate primarily at the network and transport layers (Layers 3/4). They rely on static signatures and volumetric thresholds, making them ineffective at parsing context or understanding malicious intent hidden within perfectly formatted HTTP requests. 

This project solves the challenge of Layer 7 threat detection by translating hierarchical, semi-structured API traffic into mathematical behavioral and lexical feature vectors for machine learning analysis.

## 3. Industry Relevance
APIs are currently the number one attack vector in cybersecurity. Vulnerable APIs lead directly to unauthorized data exfiltration, account takeovers, scraping, and remote system compromise. Building intelligent, adaptable detection mechanisms is critical, as attackers continuously evolve polymorphic payloads that bypass traditional, static rule-based security perimeters.

## 4. Classification Theory
* **Binary Classification:** Predicts whether traffic is simply `Normal` or `Attack`. While easier to train, it provides limited value to security teams during an active incident.
* **Multi-class Classification (Used in this project):** Categorizes traffic into specific labels. This is critical for incident response; mitigating a Denial of Service (DoS) attack requires fundamentally different infrastructure actions than patching a SQL Injection (SQLi) vulnerability.

**Target Categories:**
* Normal Traffic
* SQL Injection (SQLi)
* Cross-Site Scripting (XSS)
* Remote Code Execution (RCE)
* Directory Traversal
* Log4j Exploits
* Cookie Injection
* Log Forging

## 5. Dataset Strategy
We utilize the **ATRDF (API Threat Research Dataset Framework)** in JSON format. 

* **Why ATRDF over UNSW-NB15:** Datasets like UNSW-NB15 or CIC-IDS are heavily focused on packet-level telemetry (TCP flags, byte rates). ATRDF captures Layer 7 context—HTTP methods, endpoint URLs, nested headers, and JSON payloads—which is exactly where modern API abuse occurs.
* **Dataset Structure:** The dataset consists of arrays of nested JSON objects, primarily structured into `request` (headers, URL, method, body, attack tags) and `response` (status code, headers) blocks.

## 6. Feature Engineering
Raw text and JSON cannot be fed into ML models. We extract three distinct feature sets:

* **Request-Level Features:** Extracted directly from the HTTP structure (e.g., HTTP method one-hot encoding, response status code categories, payload byte size).
* **Pattern-Based (Lexical) Features:** Mathematical representations of the payload/URL string (e.g., Shannon entropy to detect obfuscation, special character counts like `<`, `>`, `'`, `=` indicative of injection).
* **Behavioral (Stateful) Features:** Time-series calculations grouped by unique entities (e.g., User-Agent or IP). These include Inter-Arrival Time (IAT), 60-second rolling request rates, and 60-second rolling 4xx/5xx failure rates to detect fuzzing and brute-forcing.

## 7. System Architecture
The architecture is modular and highly decoupled to support MLOps best practices:
1. **Data Ingestion Module:** Dynamically matches and loads multi-part JSON log files.
2. **Parser Module:** Safely flattens nested JSON hierarchies and extracts fallback identifiers (e.g., substituting missing IPs with User-Agent hashes).
3. **Feature Engineering Engine:** Calculates stateful and stateless features using vectorized Pandas operations.
4. **Unified Preprocessor:** A Scikit-Learn `ColumnTransformer` that handles mixed data types (Median Imputation + Standard Scaling for numerics; Frequentist Imputation + One-Hot Encoding for categoricals).
5. **Model Dispatcher:** Instantiates, trains, and saves multiple algorithmic models.
6. **Artifact Manager:** Persists trained models, the preprocessor pipeline, and evaluation metrics to disk for live inference.

## 8. Pipeline Flowchart

```text
[Raw JSON Logs] 
      │
      ▼
[JSON Parser] ──► Flattens nested requests/responses
      │
      ▼
[Feature Engine] ──► Generates Lexical & Behavioral vectors
      │
      ▼
[Stratified Split] ──► Ensures minority attacks are represented
      │
      ▼
[ColumnTransformer] ──► Imputes missing values, scales, and encodes
      │
      ▼
[Model Training] ──► Fits LR, RF, and Gradient Boosting algorithms
      │
      ▼
[Evaluation & Artifact Persistence] ──► Saves .joblib files and visualizations
```

## 9. Model Comparison

| Model | Role | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Baseline | Fast, establishes linear separability baseline, highly interpretable. | Fails against complex, non-linear behavioral traffic (high false positive rate). |
| **Random Forest** | Primary | Robust to class imbalance, handles non-linear data well, fast parallel training. | Can consume significant memory with very deep trees. |
| **Gradient Boosting** | Benchmark | Sequential error correction often yields the absolute highest accuracy. | Computationally expensive; slow sequential training time. |

## 10. Results & Key Insights
Because cybersecurity data is severely imbalanced (e.g., 99% Normal, 1% Attack), standard Accuracy is a misleading metric. We rely on **Macro F1-Score, Precision, Recall,** and **ROC-AUC**.

* **Key Findings:** Ensemble methods dramatically outperformed linear models. Random Forest and Gradient Boosting achieved a **Macro F1-score of ~0.81**, compared to Logistic Regression's 0.58. 
* **Speed vs. Performance:** Random Forest trained 30x faster than Gradient Boosting while yielding nearly identical predictive performance, making it the optimal choice for this specific pipeline.
* **The Log4j Blindspot:** All models exhibited near-zero recall for the `LOG4J` class. This highlights that purely statistical extraction (packet size, error rates) fails against highly specific string-lookup vulnerabilities. Deep NLP embeddings or regex signatures are required to patch this blindspot.

## 11. Limitations
1. **Dataset Bias:** The system's intelligence is bounded by the ATRDF dataset; it may struggle to classify novel "Zero-Day" payloads not represented in the training distribution.
2. **Stateless Limitations:** The current model evaluates requests iteratively. Complex, multi-step Business Logic Abuse (BLA) spanning days will evade detection.
3. **Compute Overhead:** Calculating rolling windows (e.g., 60-second request rates) in real-time introduces latency overhead in high-throughput production environments.
4. **Lexical Evasion:** Heavily obfuscated payloads (e.g., multi-layered Base64 encoding) can bypass basic character-count feature engineering.

## 12. Future Scope
1. **Stateful Deep Learning:** Integrating Long Short-Term Memory (LSTM) networks to evaluate sequences of API calls.
2. **Streaming Architecture:** Refactoring the inference engine to run atop Apache Kafka for real-time, low-latency stream processing.
3. **Advanced Payload Parsing:** Using NLP embeddings (like BERT) to vectorize raw request bodies to catch complex injection strings.
4. **Unsupervised Anomaly Detection:** Adding an Isolation Forest parallel pipeline to flag unknown zero-day attacks that do not match supervised labels.
5. **Automated Mitigation:** Connecting the model's confidence scores directly to an API Gateway (like Kong) to instantly drop traffic exceeding an 85% risk threshold.

## 13. Setup Instructions

**Prerequisites:** Python 3.8+

1. Clone the repository:
```bash
git clone https://github.com/yourusername/api-abuse-detection.git
cd api-abuse-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 14. How to Run

**1. Train the Pipeline & Generate Artifacts:**
Drop your raw `dataset_X_train.json` and `dataset_X_val.json` files into the `data/` directory, then run:
```bash
python src/main.py
```
*This will train the models and populate the `/artifacts` folder with saved models, preprocessors, and plots.*

**2. Run Interactive Inference (Demo Mode):**
To test the trained models against predefined edge cases or custom JSON inputs:
```bash
python demo.py
```

## 15. Example Predictions

**Sample Input (Interactive Mode):**
```json
{
  "request": {
    "method": "POST",
    "url": "/api/v1/auth",
    "headers": {"User-Agent": "curl/7.68.0"},
    "body": "{\"username\": \"admin' OR '1'='1\"}"
  }
}
```

**Console Output:**
```text
------------------------------------------------------------
📌 CUSTOM INFERENCE RESULT
------------------------------------------------------------
🛡️ Logistic Regression  | Pred: ⚠️ SQL Injection     | Conf: 88.42%
🛡️ Random Forest        | Pred: ⚠️ SQL Injection     | Conf: 96.10%
🛡️ Gradient Boosting    | Pred: ⚠️ SQL Injection     | Conf: 94.85%
------------------------------------------------------------
```

## 16. References
1. Pan, Y., Sun, F., et al. (2023). "Detecting Web Attacks Using Multi-Stage Machine Learning in API Gateways." *IEEE Transactions on Network and Service Management*.
2. Chen, X., Wang, Y., & Liu, Z. (2024). "Deep Neural Networks for Semantic Extraction and Detection of JSON Injection Vulnerabilities." *Elsevier: Computers & Security*.
3. OWASP Foundation. (2023). *OWASP API Security Top 10*. Open Web Application Security Project.
4. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*.