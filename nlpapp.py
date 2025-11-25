import streamlit as st
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Streamlit App Title
st.title(" NLP Text Processing Dashboard)")
st.write("Enter a paragraph and click a button to perform a specific NLP operation!")


# Input Text
paragraph = st.text_area("Enter your paragraph here:", height=200)

# Initialize helpers
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 1️ Tokenization
if st.button(" Generate Tokens"):
    tokens = nltk.word_tokenize(paragraph)
    st.subheader("Tokens:")
    st.write(tokens)

# 2️ Lemmatization
if st.button(" Generate Lemmas"):
    tokens = nltk.word_tokenize(paragraph)
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    st.subheader("Lemmatized Words:")
    st.write(lemmas)

# 3️ Stopword Removal
if st.button(" Remove Stopwords"):
    tokens = nltk.word_tokenize(paragraph)
    filtered = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    st.subheader("Words after Stopword Removal:")
    st.write(filtered)

# 4️ POS Tagging
if st.button(" POS Tagging"):
    tokens = nltk.word_tokenize(paragraph)
    pos_tags = nltk.pos_tag(tokens)
    st.subheader("Part of Speech Tags:")
    st.write(pos_tags)

# 5️ Named Entity Recognition (Using NLTK)
if st.button("🏷 Named Entity Recognition (NER)"):
    tokens = nltk.word_tokenize(paragraph)
    pos_tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(pos_tags, binary=False)
    named_entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity_name = " ".join(c[0] for c in chunk)
            entity_type = chunk.label()
            named_entities.append((entity_name, entity_type))
    st.subheader("Named Entities:")
    st.write(named_entities)

# 6️ Bag of Words (BoW)
if st.button(" Bag of Words (BoW) Embedding"):
    sentences = nltk.sent_tokenize(paragraph)
    corpus = []
    for s in sentences:
        review = re.sub('[^a-zA-Z]', ' ', s)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
        corpus.append(' '.join(review))
    cv = CountVectorizer()
    X_bow = cv.fit_transform(corpus).toarray()
    st.subheader("Bag of Words Matrix:")
    st.write("Shape:", X_bow.shape)
    st.write(X_bow)
    st.write("Feature Names:", cv.get_feature_names_out())

# 7️ TF-IDF Embedding
if st.button(" TF-IDF Embedding"):
    sentences = nltk.sent_tokenize(paragraph)
    corpus = []
    for s in sentences:
        review = re.sub('[^a-zA-Z]', ' ', s)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
        corpus.append(' '.join(review))
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(corpus).toarray()
    st.subheader("TF-IDF Matrix:")
    st.write("Shape:", X_tfidf.shape)
    st.write(X_tfidf)
    st.write("Feature Names:", tfidf.get_feature_names_out())

# 8️ Word2Vec (CBOW)
if st.button(" Word2Vec (CBOW)"):
    text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
    text = re.sub(r'\s+', ' ', text.lower())
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    for i in range(len(sentences)):
        sentences[i] = [word for word in sentences[i] if word not in stop_words]
    model_cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    st.subheader("Word2Vec (CBOW) Model:")
    st.write("Vocabulary Size:", len(model_cbow.wv))
    st.write("Example Similar Words for 'india':")
    try:
        st.write(model_cbow.wv.most_similar('india'))
    except:
        st.write("Word 'india' not found in vocabulary.")

# 9️ Word2Vec (Skip-Gram)
if st.button("⚙ Word2Vec (Skip-Gram)"):
    text = re.sub(r'\[[0-9]*\]', ' ', paragraph)
    text = re.sub(r'\s+', ' ', text.lower())
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    for i in range(len(sentences)):
        sentences[i] = [word for word in sentences[i] if word not in stop_words]
    model_sg = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
    st.subheader("Word2Vec (Skip-Gram) Model:")
    st.write("Vocabulary Size:", len(model_sg.wv))
    st.write("Example Similar Words for 'india':")
    try:
        st.write(model_sg.wv.most_similar('india'))
    except:
        st.write("Word 'india' not found in vocabulary.")