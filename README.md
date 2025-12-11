# Shinhancard_Oliveyoung
Measuring Mobile Promotion Effectiveness Using PSM, Heckman Selection Model, and Dragonnet

This repository contains software, code, and documentation for the research project:

“Measuring the Effectiveness of Mobile Promotions on Promotion Use and Store Spending:
A Field Experiment of Shinhan Card Mobile Promotions on the Olive Young Retail Store.”

The project analyzes large-scale field experiment data from 592,859 card users, combining traditional causal inference methods with deep-learning–based heterogeneous treatment effect models (Dragonnet).

The repository includes:

-Data preprocessing scripts

-PSM pipeline for selection-bias correction

-Heckman two-step sample-selection models

-Dragonnet implementation for estimating ITE (individual treatment effects)

-SHAP analysis for interpreting deep causal models

-Visualization and replication materials


This README summarizes the research background, datasets, methodology, and instructions for fully reproducing the analysis.

# Introduction
Traditional mobile promotion studies often measure only whether a consumer uses a promotion.
However, for transaction-based platforms (e.g., credit card companies), store spending is the core business metric.

Using Shinhan Card’s 2019 mobile promotion experiment at Olive Young, this study:

1. Analyzes two behavioral outcomes
-Promotion Use (binary)
-Store Spending (continuous)

2. Corrects two sequential selection biases that occur in real-world mobile promotions:
-Whether a user downloads/receives a promotion (self-selection)
-Whether a user visits the store to redeem it (truncation bias)

3. Compares classical causal inference (PSM + Heckman) with modern ML-based causal estimation (Dragonnet).

4. Demonstrates that mobile promotions significantly increase both promotion use and store spending, especially for messages based on loss aversion and social norms.

# Requirements and setup
A. Traditional Causal Models (PSM + Heckman)

Requirements: R >= 4.3

Packages: MatchIt/dplyr/sampleSelection/sandwich/lmtest/ggplot2

B. Dragonnet + SHAP Interpretation (Python)

Requirements: Python 3.10+

tensorflow 1.15 or 2.0-compatible build/keras == 2.2.4/numpy < 2.0/scikit-learn/pandas/matplotlib/shap

# Data
The dataset originates from Shinhan Card’s 2019 nationwide mobile promotion experiment.

Samples
-592,859 users received one of six message types over three experiment days

-Only 14,329 users (2.4%) downloaded/accepted the promotion

→ indicates strong self-selection behavior

| Category             | Examples                                        |
| -------------------- | ----------------------------------------------- |
| **Demographics**     | SEX_GB, AGE_GB                                  |
| **Usage History**    | USE_CNT_18Y, USE_AMT_18Y1, CNT_18Y, CNT_18Y_OLV |
| **Card Tenure**      | PERIOD_M                                        |
| **Message Type**     | MSG1–MSG5                                       |
| **Visit & Spending** | GAP_MIN1, GAP_MIN1_USE                          |
| **Response Timing**  | MSG_OFF_GAP                                     |
| **Coupon Download**  | OFF_YN                                          |

# Contact
For questions or collaboration:
조형찬 (Hyeongchan Cho), PhD
Kyung Hee University
yugnalgum@khu.ac.kr
