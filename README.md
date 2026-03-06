# Brugada Syndrome Detection using Machine Learning

## Competition
IDSC 2026 Machine Learning Challenge

## Authors
- Nur Insyirah Binti Azhan
- Muhammad Zuhairi Bin Ahmad Rizal

## Group
MedMetric

## Project Description
This project develops a machine learning pipeline to detect Brugada Syndrome from ECG signals using the Brugada-HUCA dataset from PhysioNet.

The system extracts clinically meaningful ECG features and trains an SVM classifier to distinguish between normal and Brugada cases.

## Dataset
Brugada-HUCA dataset from PhysioNet:

https://physionet.org/content/brugada-huca/1.0.0/

## Pipeline
1. Load ECG data from PhysioNet
2. Apply Butterworth bandpass filter
3. Detect R-peaks using Pan-Tompkins algorithm
4. Extract ECG features:
   - J-point elevation
   - QRS duration
   - T-wave amplitude
   - R-to-S amplitude drop
5. Train SVM classifier with nested cross-validation
6. Evaluate using AUC, sensitivity, specificity


