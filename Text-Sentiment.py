import re
import csv
import base64
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import SMOTE, RandomOverSampler
from streamlit_option_menu import option_menu
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as NLTKAnalyzer


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

def preproses(inputan):
    # Remove user tags
    clean_tag = re.sub('@\S+', '', inputan)
    # Remove URLs
    clean_url = re.sub('https?:\/\/.*[\r\n]*', '', clean_tag)
    # Remove hashtags
    clean_hastag = re.sub('#\S+', ' ', clean_url)
    # Remove non-alphabetic characters
    clean_symbol = re.sub('[^a-zA-Z]', ' ', clean_hastag)
    # Convert to lowercase
    casefolding = clean_symbol.lower()
    # Tokenize text
    token = word_tokenize(casefolding)
    # Remove stopwords
    listStopword = set(stopwords.words('indonesian') + stopwords.words('english'))
    stopword = [x for x in token if x not in listStopword]
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    katastem = [stemmer.stem(x) for x in stopword]
    # Join tokens to form the processed text
    joinkata = ' '.join(katastem)
    return clean_symbol, casefolding, token, stopword, katastem, joinkata

def analyze_sentiment(text, method):
    if method == 'VADERSENTIMENT':
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)
        if score['compound'] >= 0:
            return 'positif'
        else:
            return 'negatif'
    elif method == 'TEXTBLOB':
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positif'
        else:
            return 'negatif'
    elif method == 'NLTK':
        nltk_analyzer = NLTKAnalyzer()
        score = nltk_analyzer.polarity_scores(text)
        if score['compound'] >= 0:
            return 'positif'
        else:
            return 'negatif'
        

# Sidebar Pre-Processing
with st.sidebar:
    st.markdown("""
        <h1 style='text-align: center; color: white; font-size: 36px; font-weight: bold;'>
            ANALISIS TEKS SENTIMEN
        </h1>
        <br>
    """, unsafe_allow_html=True)
    option = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Pre Processing - Import dataset", "TF-IDF","IMPLEMENTASI", ],  # required
        icons=["house", "file-earmark-spreadsheet", "bar-chart", "file-text"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

if option == 'Home':
    Home, Laporan = st.tabs(['Home', 'Laporan']) # Ambil tab pertama

    with Home:
        st.title("ANALISIS SENTIMEN KOMENTAR YOUTUBE")
        st.subheader('Disusun Oleh:')
        st.subheader('Kelompok A4')
        st.text("""
                    1. I Wayan Adhi Surya Gemilang	(2108561039)
                    2. I Putu Herdy Juniawan        (2208561033)
                    3. Azra Aaliyah Seisha Sybille  (2208561040)
                    4. I Agus Indra Dipta Prayoga	(2208561114)
                    5. Kadek Wina Septhiana	        (2208561132)

                    """)
    
    with Laporan:
        st.title("LINK LAPORAN TUGAS PPDM")
        st.markdown("""
        <a href="https://docs.google.com/document/d/1uX5gz7I1r6AP77g3pbdyQFH06NlTN9yTnXB69WXiEOY/edit" target="_blank">
            <button style="background-color:green;color:white;padding:10px;border:none;border-radius:5px;">
                Klik Disini
            </button>
        </a>
        """, unsafe_allow_html=True)


elif option == 'IMPLEMENTASI':
    st.title("ANALISIS KOMENTAR")
    inputan = st.text_input(label="**Masukan Komentar** 👇 ")
    method = st.selectbox("Pilih Library untuk Analisis Sentimen", ("VADERSENTIMENT", "TEXTBLOB", "NLTK"))

    if st.button("Submit"):
        with st.spinner('Melakukan Stemming'):
            # Preprocess input text
            clean_symbol, casefolding, token, stopword, katastem, joinkata = preproses(inputan)

            # Predict the sentiment
            X_pred = analyze_sentiment(joinkata, method)
            
            # Display the result
            hasil = f"Berdasarkan data yang Anda masukkan, maka ulasan masuk dalam kategori: {X_pred}"
            if X_pred == 'positif':
                st.success(hasil)
            else:
                st.warning(hasil)
            
            # Display preprocessing steps
            st.subheader('Preprocessing')
            st.write('Cleansing:', clean_symbol)
            st.write("Case Folding:", casefolding)
            st.write("Tokenisasi:", token)
            st.write("Stopword:", stopword)
            st.write("Stemming:", katastem)
            st.write("Siap Proses:", joinkata)
            
            # Show balloons effect
            st.balloons()

elif option == 'Pre Processing - Import dataset':
    st.title("DATA UNDERSTANDING")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(10))
        st.write(df.info())
        st.write(df.describe())
        st.write("Baris, Kolom")
        st.write(df.shape)
        st.write("Menghilangkan data yg kosong")
        st.write(df.dropna(inplace=True))
        st.write(df.isnull().sum())

        kolom_komentar = st.text_input("Masukkan nama kolom komentar:")

        # Tombol untuk tahap Pre-Processing
        if st.button('Lanjut Tahap PRE-PROCESSING'):
            # Meminta pengguna untuk memasukkan nama kolom Komentar

            st.title("PRE-PROCESSING")
            st.subheader("a. Case Folding")
            # Case Folding: Mengubah semua teks menjadi huruf kecil
            df['komen case folding'] = df[kolom_komentar].str.lower()

            # Menampilkan hasil Case Folding
            st.subheader("Data Setelah Case Folding")
            st.write(df.shape)
            st.write(df.head(10)[['komen case folding']])

            st.subheader("b. Tokenizing")
            df = df.dropna(subset=['komen case folding'])
            # Tokenizing: proses menguraikan suatu teks atau dokumen menjadi unit-unit yang lebih kecil yang disebut dengan token
            df['komen tokenized'] = df['komen case folding'].apply(word_tokenize)

            st.subheader("Data Setelah Tokenizing:")
            st.write(df.shape)
            st.write((df.head(10)[['komen tokenized']]))

            st.subheader("c. Normalization (Stopward Removal)")

            # Normalization: Hapus karakter khusus, angka, dan kata-kata berhenti
            stop_words = set(stopwords.words('indonesian'))
            df['komen normalized'] = df['komen tokenized'].apply(lambda tokens: [word for word in tokens if word.isalpha() and word not in stop_words])
            st.subheader("Data Setelah Normalization:")
            st.write(df.shape)
            st.write((df.head(10)[['komen normalized']]))

            st.subheader("d. Stemming")
            with st.spinner('Melakukan Stemming') :
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                df['komen stemming'] = df['komen normalized'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])
                st.subheader("Data Setelah Stemming:")
                st.write(df.shape)
                st.write(df.head(10)[['komen stemming']])

            st.subheader("Dataset baru yang telah dibersihkan")
            df['komen bersih'] = df['komen normalized'].apply(lambda tokens: ' '.join([stemmer.stem(word) for word in tokens]))
            st.write("\nData Clean:")
            st.write(df.shape)
            st.write(df.head(10)[['komen bersih']])

            df = df[df["SENTIMEN"] != "Netral"]


            st.subheader("Data Dengan Label Positif dan Negatif saja")
            st.write(df.shape)
            st.write(df)
            st.subheader("Download Dataset yang berlabel Positif dan Negatif saja")
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_dataset.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

elif option == 'TF-IDF':
    st.title("1. TF-IDF")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head(10))
        st.write("(Baris, Kolom)")
        st.write(df.shape)
        komen_bersih = st.text_input("Masukkan nama kolom Komentar Bersih :")

        if komen_bersih in df.columns:  # Pastikan nama kolom komentar valid
            # Membersihkan nilai NaN
            df.dropna(subset=[komen_bersih], inplace=True)
            
            cv = CountVectorizer()
            term_fit = cv.fit(df[komen_bersih].astype(str))  # Pass the column content, converted to strings
            st.write("Jumlah Kata Komentar Bersih: ")
            st.write(len(term_fit.vocabulary_))
            st.write("Kamus Kata:")
            st.write(term_fit.vocabulary_)

            # Extract the comment at index 1
            if len(df) > 1:
                komen_index_1 = df[komen_bersih].iloc[1]
                st.write(f"Komentar pada index 1: {komen_index_1}")

                # Tokenize the comment and pick the first word
                first_word = komen_index_1.split()[0]
                st.write(f"Kata pertama pada index 1: {first_word}")

                # Transform the first word using the fitted vectorizer
                first_word_vector = cv.transform([first_word])
                st.write(first_word_vector)
            else:
                st.error("DataFrame doesn't have enough rows to access index 1.")

            # Transform the entire comment at index 1
            term_frequency = term_fit.transform([komen_index_1])  # Transform as a single document list
            st.write("Mengubah menjadi daftar dokumen tunggal :")
            st.write(term_frequency)

            # Fit and transform using TfidfTransformer
            tfidf_transformer = TfidfTransformer()
            tfidf_transformer.fit(term_frequency)
            st.write("Transformasi menggunakan TfidfTransformer :")
            st.write(tfidf_transformer.idf_)

            tfidf = tfidf_transformer.transform(term_frequency)
            st.write(tfidf)

            st.title("2. NLP")
            df_label = df[["COMMENT","komen bersih", "SENTIMEN"]]
            SENTIMEN = st.text_input("Masukkan nama kolom label:")
            if SENTIMEN in df.columns:
                sentimen_df = pd.value_counts(df[SENTIMEN], sort=True)
                sentimen_df.plot(kind='bar', color=["green", "red"])
                plt.title('Sentimen Ulasan Positif dan Negatif')
                st.pyplot(plt)
            else:
                st.error("Kolom Label tidak ditemukan di dalam DataFrame.")
            
            st.write("Mengubah Sentimen Positif menjadi 1 dan Negatif menjadi 0 :")
            df = df.replace({'Positif':1, 'Negatif':0})
            st.write(df[['SENTIMEN','komen bersih']])

            st.title("3. VISUALISASI")
            data_negatif = df[df[SENTIMEN] == 0]
            data_positif = df[df[SENTIMEN] == 1]

            st.subheader("Visualisasi Data Negatif")
            all_text_s0 = ' '.join(word for word in data_negatif[komen_bersih].astype(str))  # Pastikan data diubah menjadi string
            wordcloud = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s0)
            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Visualisasi Comment Negatif")
            plt.margins(x=0, y=0)
            st.pyplot(plt)

            st.subheader("Visualisasi Data Positif")
            all_text_s1 = ' '.join(word for word in data_positif[komen_bersih].astype(str))
            wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s1)
            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Visualisasi Comment Positif")
            plt.margins(x=0, y=0)
            st.pyplot(plt)

            X_train, X_test, y_train, y_test = train_test_split(df['komen bersih'], df['SENTIMEN'],
            test_size=0.2, stratify=df_label['SENTIMEN'], random_state=30)
            
            st.title("4. DATA PREPARATION")
            st.subheader("bagi 20 persen data testing, 80 persen data training")
        
            # Memastikan kolom komentar diubah menjadi string
            X_train = X_train.astype(str)
            X_test = X_test.astype(str)

            vectorizer = TfidfVectorizer(decode_error='replace', encoding='utf-8')
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

            st.write("80 persen data training :")
            st.write(X_train.shape)
            st.write("20 persen data testing :")
            st.write(X_test.shape)
            
            X_train = X_train.toarray()
            X_test = X_test.toarray()

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            # Contoh data hasil resampling
            sentimen_counts = y_train_resampled.value_counts()

            plt.figure(figsize=(5, 3))
            sns.countplot(x=y_train_resampled, palette={"red", "lightskyblue"})
            plt.title('Visualisasi Sentimen Menggunakan SMOTE')
            plt.xlabel('SENTIMEN')
            plt.ylabel("Jumlah")
            plt.xticks(sentimen_counts.index, ['0', '1'])
            st.pyplot(plt)

            st.title("5. Modeling")
            naive_bayes = MultinomialNB()
            naive_bayes.fit(X_train_resampled, y_train_resampled)

            y_pred = naive_bayes.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, target_names=["NEGATIF", "POSITIF"])

            st.write("Akurasi Model Naive Bayes : ", accuracy)

            st.write("Laporan Klasifikasi :", classification_rep)
            st.title("6. Tuning")

            k = 5
            kf = KFold(n_splits=k, shuffle=True, random_state=42)

            # Contoh data fitur dan label
            X = np.random.rand(100, 10)
            y = np.random.choice([0, 1], 100)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Resample data (misal menggunakan RandomOverSampler)
            ros = RandomOverSampler(random_state=42)
            x_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

            # Pastikan data resampled adalah array numpy
            x_train_resampled = np.array(x_train_resampled)
            y_train_resampled = np.array(y_train_resampled)

            # Definisikan hyper-parameter
            param_grid = {
                'alpha': [0.1, 0.5, 1.0],
                'fit_prior': [True, False]
            }

            # Inisialisasi model Naive Bayes
            naive_bayes = MultinomialNB()

            # Inisialisasi GridSearchCV dengan model, grid hyper-parameter, dan k-fold cross validation
            grid_search = GridSearchCV(estimator=naive_bayes, param_grid=param_grid, cv=k, scoring='accuracy')

            # Streamlit interface
            st.subheader("Hyper-Parameter Tuning untuk Model Naive Bayes")

            # Pencarian hyper-parameter terbaik
            grid_search.fit(x_train_resampled, y_train_resampled)

            # Hyper-parameter terbaik dan skor akurasi terbaik
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            st.write("Hyper-parameter terbaik:", best_params)
            st.write("Skor akurasi terbaik:", best_score)

            # Prediksi dengan model terbaik
            best_model = grid_search.best_estimator_
            y_pred_best = best_model.predict(X_test)

            # Hitung dan cetak akurasi model terbaik
            accuracy_best = accuracy_score(y_test, y_pred_best)
            st.write("Akurasi Model Naive Bayes dengan Hyper-parameter Terbaik:", accuracy_best)

            # Cetak laporan klasifikasi menggunakan model terbaik
            classification_rep_best = classification_report(y_test, y_pred_best, target_names=["NEGATIF", "POSITIF"])
            st.write("Laporan Klasifikasi dengan Hyper-parameter Terbaik:")
            st.text(classification_rep_best)

            # K-Fold Cross Validation dengan model terbaik
            accuracies = []  # Array akurasi dari setiap fold
            for train_index, test_index in kf.split(x_train_resampled):
                X_train_fold, X_test_fold = x_train_resampled[train_index], x_train_resampled[test_index]
                y_train_fold, y_test_fold = y_train_resampled[train_index], y_train_resampled[test_index]

                best_model.fit(X_train_fold, y_train_fold)  # Latih model menggunakan best model saat hypertuning
                y_pred_fold = best_model.predict(X_test_fold)  # Prediksi menggunakan data uji fold

                accuracy_fold = accuracy_score(y_test_fold, y_pred_fold)  # Hitung akurasi
                accuracies.append(accuracy_fold)

            # Hitung rata-rata akurasi dari semua fold
            mean_accuracy = np.mean(accuracies)
            st.write("Akurasi Model Naive Bayes dengan K-Fold Cross Validation:", mean_accuracy)

            # Cetak akurasi tiap fold
            fold_number = 1
            for accuracy_fold in accuracies:
                st.write(f"Akurasi Fold {fold_number}: {accuracy_fold}")
                fold_number += 1

            # Menampilkan data dummy sebagai placeholder
            st.write("Contoh Data:")
            st.write(X[:5])  # Menampilkan 5 baris pertama dari data fitur
            st.write("Label:")
            st.write(y[:5])  # Menampilkan 5 label pertama
        else:
            st.error("Nama kolom komentar tidak valid. Silakan coba lagi.")
