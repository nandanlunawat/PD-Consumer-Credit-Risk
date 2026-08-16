# Probability of Default Modelling for Consumer Credit Risk

## Overview

This project builds an interpretable Probability of Default (PD) scorecard for consumer loans, using LendingClub data from 2007–2018. The objective is to estimate, at the point of application, how likely a borrower is to default over the life of the loan — using only information a lender would actually hold at that moment. The emphasis throughout is on methodological correctness: outcome observability, leakage control and out-of-time validation, rather than headline accuracy.

## Business Problem

Every consumer loan is a decision made under uncertainty: approve and risk a credit loss, or decline and lose the customer. A PD model quantifies that risk per applicant so lending decisions can be consistent, reviewable and aligned to risk appetite. Because credit decisions must be explainable to customers and regulators, the model also has to be interpretable — a constraint that shapes the modelling choices below.

## Methodology

**Modelling universe and observability.** The raw file contains 2,260,701 loans, but later vintages are right-censored: their outcomes were not yet observable at the March 2019 data cutoff. Loans issued in 2017–2018 that appear "resolved" are disproportionately early prepayments and early defaults, which makes them a biased test population. The modelling universe is therefore restricted to loans whose entire contractual term completed before the cutoff — 779,026 fully observed loans (bad rate 15.19%). The primary model covers the 36-month product (710,807 loans); fully observed 60-month history predates any honest out-of-time window, so that product is out of scope.

**Target definition.** Bad = Charged Off or Default over the full term; Good = Fully Paid. Loans still unresolved (Current, Late, In Grace Period) are excluded rather than guessed. No default dates are assumed — every label reflects an observed terminal outcome.

**Leakage screening.** All candidate variables were audited before modelling. Post-origination fields (payments, recoveries, settlements, hardship, outstanding principal) were removed as target leakage. LendingClub's own risk outputs (grade, sub-grade, interest rate, installment) were also removed, so the model is an independent view of borrower risk built from application-time characteristics, not a reconstruction of the platform's pricing.

**Feature selection and transformation.** Numeric features were binned into deciles and all features transformed to Weight of Evidence (WOE), with a 0.5 adjustment for zero-count bins and missing values held in their own bin. Bin edges, WOE values and Information Value were learned on the training set only and applied unchanged out-of-time. Twelve features with IV ≥ 0.02 entered the final model. An engineered installment-to-income ratio was rejected during development after a coefficient sign flip exposed collinearity and interest-rate contamination.

**Champion and challenger.** The champion is a logistic regression on WOE features — chosen because credit decisioning requires coefficient-level interpretability. A gradient boosting challenger (fixed hyperparameters, full candidate feature set) was benchmarked against it; its ~0.024 Gini uplift was judged insufficient to give up interpretability, noting the comparison is directional rather than perfectly controlled.

**Out-of-time validation.** The model was trained on 2012–2014 originations (306,462 loans, bad rate 13.25%) and evaluated on strictly later 2015–Q1 2016 originations (372,811 loans, bad rate 15.18%) — the way credit models are validated in practice, because the production question is performance on future applicants.

## Model Validation

| Metric (out-of-time, 2015–Q1 2016) | Logistic champion | Gradient boosting challenger |
|---|---:|---:|
| AUC | 0.671 | 0.683 |
| Gini | 0.342 | 0.366 |
| KS | 0.247 | 0.264 |

Discrimination was stable across test vintages (Gini 0.330 → 0.342 → 0.359 from 2015H1 to 2016Q1). Predicted PDs ran below observed bad rates by up to ~3 percentage points in the highest decile, consistent with base-rate drift between training and test vintages; intercept recalibration is documented as the production remedy rather than applied.

## Risk Segmentation

Scored out-of-time borrowers were segmented into four PD bands (40/30/20/10 population split). Observed bad rates are strictly monotonic across bands:

| Band | Population | Observed bad rate |
|---|---:|---:|
| Low | 40% | 7.86% |
| Medium | 30% | 15.23% |
| High | 20% | 21.89% |
| Very High | 10% | 30.89% |

The two highest-risk bands — 30% of borrowers — account for **49.2% of all observed defaults**, which is the concentration that makes the segmentation usable for decisioning.

## Key Credit-Risk Drivers

The strongest application-time drivers of default, by Information Value and model contribution: FICO score, accounts opened in the past 24 months, inquiry recency and count, revolving utilisation, debt-to-income, loan-to-income and loan purpose.

## Business Application

The framework could support risk-based approval cutoffs (for example, routing the Very High band to manual review), risk-based pricing tiers, and portfolio monitoring through shifts in band mix across vintages. No claim is made that the model was deployed or changed any lending decision.

## Limitations

1. Marketplace-lending data rather than proprietary bank underwriting data — population and process differ.
2. The full-observability rule limits usable vintages to Q1 2016 and earlier; the model is validated on a pre-2017 lending environment.
3. Scope is the 36-month product only.
4. The target is lifetime default over the loan term, not a fixed-horizon 12-month PD — the dataset contains no default dates.
5. Out-of-time PDs show calibration drift from the vintage base-rate shift.
6. No macroeconomic variables are included, and discriminatory performance alone does not establish economic value.

## Reproducibility

The full pipeline — universe construction, target definition, leakage screening, WOE/IV, model fitting and validation — runs top to bottom in a single notebook. The dataset (~1.3 GB) is downloaded at runtime via `kagglehub` and is not included in this repository; a Kaggle API token is required. The universe-build step takes roughly 10–15 minutes; results are deterministic (fixed random seed, deterministic time-based split).

```
pip install -r requirements.txt
jupyter notebook PD_Model_Consumer_Credit_Risk.ipynb
```
