🦵 Knee Health Monitoring System (ML-Based)
📌 Overview

This project focuses on building a machine learning–based knee health monitoring system that analyzes sequential data to detect and evaluate knee health conditions. Multiple deep learning architectures were implemented, compared, and evaluated to identify the most efficient and accurate model. The final selected model was deployed on Amazon EC2 for scalable inference.

🧠 Models Implemented

The following deep learning model combinations were explored and evaluated:

1D CNN + LSTM

Used 1D Convolutional Neural Networks for feature extraction

Followed by LSTM for capturing temporal dependencies

1D CNN + GRU

CNN for feature extraction

GRU for sequence modeling with fewer parameters

LSTM + MLP

LSTM for temporal learning

MLP for final classification

📊 Model Comparison & Evaluation

Conducted a comparative analysis based on:

Accuracy

Training time

Parameter count

Computational complexity

GRU-based architecture outperformed others, delivering:

Comparable or better accuracy

Fewer parameters

Lower time complexity

Faster convergence

➡️ GRU was selected as the final model due to its efficiency and performance.

☁️ Deployment

The best-performing model (GRU-based) was deployed on Amazon EC2

Enabled scalable and continuous inference for real-time knee health monitoring

Cloud deployment ensures availability and production readiness

🛠️ Tech Stack

Languages: Python

Deep Learning: LSTM, GRU, CNN (1D), MLP

Frameworks: TensorFlow

Cloud: Amazon EC2

Evaluation: Accuracy, performance comparison, time complexity analysis

🎯 Key Highlights

Implemented and compared multiple deep learning architectures

Optimized model selection based on performance and efficiency

Deployed a production-ready ML model on cloud infrastructure

Focused on real-world applicability and scalability

🚀 Future Enhancements

Integration with real-time sensor data

Web or mobile interface for live monitoring

Model optimization for edge deployment

Automated retraining pipeline

👤 Author

Bhupesh Bhatia
