# YT-Comment-Sentiment-NaiveBayes

Sentiment analysis of YouTube comments using Natural Language Processing (NLP) techniques and the Multinomial Naïve Bayes algorithm. This project focuses on classifying comments related to boxing matches into positive and negative sentiments through an interactive Streamlit application.

## Features

* Import and process YouTube comment datasets
* Text preprocessing pipeline:

  * Cleansing
  * Case folding
  * Tokenization
  * Stopword removal
  * Stemming using Sastrawi
* TF-IDF feature extraction
* Data balancing using SMOTE
* Sentiment classification with Multinomial Naïve Bayes
* Hyperparameter tuning using GridSearchCV
* Model evaluation using multiple metrics
* Interactive web interface built with Streamlit

## Dataset

The dataset contains YouTube comments and their corresponding sentiment labels.

| USERNAME         | COMMENT                                                      | SENTIMENT |
| ---------------- | ------------------------------------------------------------ | --------- |
| @indriSaman      | Pantas menang El Rumi, lebih cerdik El Rumi.                 | Positive  |
| @mtaqiyuddin4910 | Hahah penilaian konyol.                                      | Negative  |
| @user-cv7qm7ld8m | Heleeeehhhh jeprii ninju angin doank.                        | Negative  |
| @smsun1030       | El Rumi masih bisa berkembang pesat jika dibina dengan baik. | Positive  |

## Workflow

```text
Data Collection
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Train-Test Split
    ↓
SMOTE Oversampling
    ↓
Naïve Bayes Classification
    ↓
Hyperparameter Tuning
    ↓
Model Evaluation
```

## Technologies

* Python
* Streamlit
* Pandas
* NumPy
* NLTK
* Sastrawi
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib
* Seaborn
* WordCloud
* TextBlob
* VaderSentiment

## Installation

Clone the repository:

```bash
git clone https://github.com/HrdyXD/YT-Comment-Sentiment-NaiveBayes.git
cd YT-Comment-Sentiment-NaiveBayes
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

```text
YT-Comment-Sentiment-NaiveBayes/
│
├── dataset/
│   └── sample_dataset.csv
├── app.py
├── requirements.txt
└── README.md
```

## License

This project was developed for educational and research purposes.
