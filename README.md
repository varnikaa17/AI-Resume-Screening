# AI Resume Screening System

An AI-powered resume screening prototype that compares resumes with job descriptions using TF-IDF text features, Logistic Regression, and resume–job similarity.

## Features

- Upload resumes in PDF and DOCX formats
- Extract resume text automatically
- Compare resumes with job descriptions
- Calculate an experimental model match score and text similarity
- Identify matched and potentially missing skills
- Evaluate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Visualize model performance using ROC curves

## Tech Stack

- Python 3.13
- Pandas and NumPy
- Scikit-learn
- TF-IDF and Logistic Regression
- Streamlit
- PyMuPDF and python-docx
- Matplotlib

## Project Structure

```text
Resume-Screening-project/
├── app.py
├── create_dataset.py
├── check_dataset.py
├── preprocess.py
├── split_dataset.py
├── tfidf_features.py
├── tfidf_similarity.py
├── train_baseline.py
├── train_similarity_model.py
├── check_leakage.py
├── cross_validate_model.py
├── requirements.txt
├── data/
├── models/
├── results/
└── README.md
```

## Setup

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Generate the dataset and run the data preparation scripts in the required order.
5. Train the models to generate the required local model files.
6. Start the application:

```bash
python -m streamlit run app.py
```

## Model Evaluation

The project evaluates models using Accuracy, Precision, Recall, F1-score, and ROC-AUC. It also includes a five-fold cross-validation script and a dataset similarity check.

Record the final cross-validation results here after verifying them.

## Limitations

The current dataset is synthetic. High evaluation scores on this dataset do not establish real-world hiring accuracy. The model score is experimental and is not a calibrated probability of job success.

This prototype should support human review rather than automatically reject or select candidates. Resume information should be handled responsibly and securely.

## Future Improvements

- Evaluate on a realistic, independently verified resume dataset
- Compare TF-IDF with transformer-based embeddings
- Improve skill extraction and explainability
- Test generalization across different job roles and datasets
- Add automated tests and deployment configuration
