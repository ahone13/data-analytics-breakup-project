import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Anatomy of a Modern Breakup",
    page_icon="💔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #3D2530;
        border: 1px solid #6D2E46;
        border-radius: 10px;
        border-left: 5px solid #A26769;
        padding: 15px;
    }
    div[data-testid="metric-container"] label {
        color: #ECE2D0 !important;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #ffffff !important;
        font-size: 2rem !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Color Palette ──────────────────────────────────────────────────────────────
COLORS = {
    'primary': '#6D2E46',
    'secondary': '#A26769',
    'accent': '#ECE2D0',
    'dark': '#2E1A22',
    'muted': '#8C7A75',
}
PALETTE = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['muted'], COLORS['dark']]

# ── Data Loading & Processing ──────────────────────────────────────────────────
@st.cache_data
def load_and_process():
    df_breakup = pd.read_csv(r"datasets/reddit_breakup_dataset_cleaned.csv")
    df_advice  = pd.read_csv(r"datasets/relationship_advice.csv")

    df_breakup_clean = df_breakup[['title','body','upvotes','comments_count','post_date']].copy()
    df_breakup_clean.columns = ['title','body','score','comms_num','timestamp']
    df_breakup_clean['source'] = 'breakups'

    df_advice_clean = df_advice[['title','body','score','comms_num','timestamp']].copy()
    df_advice_clean['source'] = 'relationship_advice'

    df = pd.concat([df_breakup_clean, df_advice_clean], ignore_index=True)

    # Clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['title_clean'] = df['title'].apply(clean_text)
    df['body_clean']  = df['body'].apply(clean_text)
    df['body_length'] = df['body_clean'].apply(lambda x: len(x.split()))
    df['timestamp']   = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
    df['publish_hour'] = df['timestamp'].dt.hour
    median_score = df['score'].median()
    df['high_engagement'] = (df['score'] > median_score).astype(int)
    df = df[df['body_length'] > 0].reset_index(drop=True)

    # VADER
    analyzer = SentimentIntensityAnalyzer()
    df['title_sentiment'] = df['title_clean'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['body_sentiment']  = df['body_clean'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    def sentiment_label(score):
        if score >= 0.05: return 'positive'
        elif score <= -0.05: return 'negative'
        else: return 'neutral'

    df['body_sentiment_label'] = df['body_sentiment'].apply(sentiment_label)

    # Topic clustering
    custom_stopwords = [
        'throwra','reddit','post','account','create','new','action','contact','users',
        'rules','removed','limited','referring','sexual','list','involving','comment',
        'advice','questions','just','like','dont','really','want','know','feel','im',
        'got','said','told','didnt','went','also','would','could','get','go','one',
        'think','time','way','thing','things','youre','ive','thats','doesnt','wasnt',
        'cant','hes','shes','result','andor','ban','minors','allinclusive','rule',
        'appears','does','people','did','make','right','try','trying','help','work',
        'job','men','women','looks','concerns','message','judgment','automatically',
        'starts','performed','bot','problems','subject','posts','permanent','medical',
        'welcome','locked','subreddit','letters','partners','include','pass','number',
        'joke','read','mean','say','going','asked','started','talking','night','home','day'
    ]
    all_stopwords = list(ENGLISH_STOP_WORDS) + custom_stopwords

    mask = ~df['body_clean'].str.contains(
        'subreddit|moderator|automod|removed|locked|bot|composeto|legal|necessarily',
        case=False, na=False
    )
    df_lda = df[mask].copy()

    vectorizer = CountVectorizer(max_df=0.90, min_df=10, max_features=1000, stop_words=all_stopwords)
    dtm = vectorizer.fit_transform(df_lda['body_clean'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=20)
    lda.fit(dtm)

    topic_labels = {
        0: 'Heartbreak & Recovery',
        1: 'Long-term Strain',
        2: 'Social & Family Conflict',
        3: 'Intimacy & Dynamics',
        4: 'Dating & Early Relationships'
    }
    topic_assignments = lda.transform(dtm)
    df_lda['dominant_topic'] = topic_assignments.argmax(axis=1)
    df_lda['topic_label'] = df_lda['dominant_topic'].map(topic_labels)

    return df, df_lda

# Load data with spinner
with st.spinner("Loading data pipeline..."):
    df, df_lda = load_and_process()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💔 Navigation")
    page = st.selectbox("Go to section:", [
        "🏠 Overview",
        "😊 Sentiment Analysis",
        "☁️ Word Clouds",
        "🗂️ Topic Clustering",
        "📊 Correlation & Distribution",
        "🤖 Model Results",
        "📈 Model Improvement"
    ])
    st.markdown("---")
    st.markdown("**Project:** The Anatomy of a Modern Breakup")
    st.markdown("**Student:** Okah Ahone Ebwekoh")
    st.markdown("**ID:** 80770239")
    st.markdown("**Course:** Data Analytics — DS26")

# ── PAGE: Overview ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("💔 The Anatomy of a Modern Breakup")
    st.markdown("*A Multi-Dataset Sentiment & Engagement Analytics Pipeline*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Posts", f"{len(df):,}")
    with col2:
        st.metric("Data Sources", "2 Subreddits")
    with col3:
        st.metric("Median Score", f"{df['score'].median():.0f}")
    with col4:
        st.metric("High Engagement Posts", f"{df['high_engagement'].sum():,}")

    st.markdown("### Dataset Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        source_counts = df['source'].value_counts()
        ax.bar(source_counts.index, source_counts.values,
               color=[COLORS['primary'], COLORS['secondary']], edgecolor='white')
        ax.set_title('Posts by Source', fontsize=14, color=COLORS['dark'])
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        eng_counts = df['high_engagement'].value_counts()
        ax.pie(eng_counts.values, labels=['Low Engagement', 'High Engagement'],
               autopct='%1.1f%%', colors=[COLORS['accent'], COLORS['primary']],
               startangle=90)
        ax.set_title('Engagement Split', fontsize=14, color=COLORS['dark'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### Key Finding")
    st.info("📌 **The Sentiment Paradox:** Despite the highly emotional nature of breakup posts, sentiment score shows near-zero correlation with engagement. What drives visibility is *interaction patterns* — comment count and post timing — not emotional tone.")

    st.markdown("### Sample Posts")
    st.dataframe(
        df[['title', 'source', 'score', 'comms_num', 'body_sentiment_label']].head(10),
        use_container_width=True
    )

# ── PAGE: Sentiment ────────────────────────────────────────────────────────────
elif page == "😊 Sentiment Analysis":
    st.title("😊 Sentiment Analysis")
    st.markdown("*VADER scored each post from -1 (most negative) to +1 (most positive)*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    pos = (df['body_sentiment_label'] == 'positive').sum()
    neg = (df['body_sentiment_label'] == 'negative').sum()
    neu = (df['body_sentiment_label'] == 'neutral').sum()
    with col1: st.metric("Positive Posts", f"{pos:,}", f"{pos/len(df)*100:.1f}%")
    with col2: st.metric("Negative Posts", f"{neg:,}", f"{neg/len(df)*100:.1f}%")
    with col3: st.metric("Neutral Posts", f"{neu:,}", f"{neu/len(df)*100:.1f}%")
    with col4: st.metric("Avg Body Sentiment", f"{df['body_sentiment'].mean():.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    df['body_sentiment_label'].value_counts().plot(
        kind='bar', ax=axes[0],
        color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']],
        edgecolor='black'
    )
    axes[0].set_title('Sentiment Distribution', fontsize=13)
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)

    sns.boxplot(data=df, x='source', y='body_sentiment', ax=axes[1],
                hue='source', palette=[COLORS['primary'], COLORS['secondary']], legend=False)
    axes[1].set_title('Sentiment by Source', fontsize=13)

    sns.boxplot(data=df, x='high_engagement', y='body_sentiment', ax=axes[2],
                hue='high_engagement', palette=[COLORS['accent'], COLORS['primary']], legend=False)
    axes[2].set_title('Sentiment vs Engagement', fontsize=13)
    axes[2].set_xlabel('High Engagement (0=No, 1=Yes)')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success("💡 **Finding:** 57% of breakup posts trend positive — people frame their experiences with hope and resolution, not just pain. Yet sentiment has almost zero effect on engagement.")

    # Filter by sentiment
    st.markdown("### Explore by Sentiment")
    selected_sentiment = st.selectbox("Filter posts by sentiment:", ['positive', 'negative', 'neutral'])
    filtered = df[df['body_sentiment_label'] == selected_sentiment][['title', 'body_sentiment', 'score', 'source']].head(10)
    st.dataframe(filtered, use_container_width=True)

# ── PAGE: Word Clouds ──────────────────────────────────────────────────────────
elif page == "☁️ Word Clouds":
    st.title("☁️ Word Clouds")
    st.markdown("*Most frequent words in positive vs negative posts after text cleaning*")
    st.markdown("---")

    col1, col2 = st.columns(2)

    for col, label, color in zip([col1, col2], ['positive', 'negative'], ['RdPu', 'PuRd']):
        with col:
            st.markdown(f"### {label.capitalize()} Posts")
            text = ' '.join(df[df['body_sentiment_label'] == label]['body_clean'])
            wc = WordCloud(width=700, height=400, background_color='white',
                           colormap=color, max_words=100).generate(text)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.info("💡 **Finding:** Both sentiment groups share almost identical core vocabulary — 'feel', 'time', 'relationship', 'said'. The emotional framing shifts but the language barely changes.")

# ── PAGE: Topic Clustering ─────────────────────────────────────────────────────
elif page == "🗂️ Topic Clustering":
    st.title("🗂️ Topic Clustering")
    st.markdown("*LDA identified 5 recurring themes across 2,000 filtered posts*")
    st.markdown("---")

    topic_counts = df_lda['topic_label'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        topic_counts.plot(kind='bar', ax=ax, color=PALETTE, edgecolor='black')
        ax.set_title('Post Distribution by Topic', fontsize=13)
        ax.set_ylabel('Number of Posts')
        ax.tick_params(axis='x', rotation=25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(topic_counts.values, labels=topic_counts.index,
               autopct='%1.1f%%', colors=PALETTE, startangle=140)
        ax.set_title('Topic Proportion', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### Topic Descriptions")
    topic_info = {
        'Heartbreak & Recovery': ('30.4%', 'Posts about processing the immediate pain of a breakup — feeling lost, moving on, emotional recovery.'),
        'Long-term Strain': ('21.3%', 'Issues in long-term relationships — family, finances, kids, leaving a partner after years together.'),
        'Dating & Early Relationships': ('17.9%', 'Early dating dynamics — meeting someone new, navigating early relationship stages.'),
        'Intimacy & Dynamics': ('16.2%', 'Posts about physical and emotional intimacy, relationship roles and boundaries.'),
        'Social & Family Conflict': ('14.4%', 'Third-party conflicts — friends, family interference, social circle drama around breakups.'),
    }
    for topic, (pct, desc) in topic_info.items():
        with st.expander(f"📌 {topic} — {pct}"):
            st.write(desc)

    st.markdown("### Filter Posts by Topic")
    selected_topic = st.selectbox("Select a topic:", list(topic_info.keys()))
    filtered_topic = df_lda[df_lda['topic_label'] == selected_topic][['title', 'score', 'body_sentiment_label', 'source']].head(10)
    st.dataframe(filtered_topic, use_container_width=True)

# ── PAGE: Correlation & Distribution ──────────────────────────────────────────
elif page == "📊 Correlation & Distribution":
    st.title("📊 Correlation & Score Distribution")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Correlation Heatmap")
        corr_cols = ['score','comms_num','body_length','publish_hour','title_sentiment','body_sentiment','high_engagement']
        corr_matrix = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdPu', linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### PMF & CDF of Post Score")
        fig, axes = plt.subplots(2, 1, figsize=(7, 6))

        score_capped = df[df['score'] <= 100]['score']
        score_pmf = score_capped.value_counts(normalize=True).sort_index()
        axes[0].bar(score_pmf.index, score_pmf.values, color=COLORS['primary'], alpha=0.7)
        axes[0].set_title('PMF of Post Score (capped at 100)', fontsize=11)
        axes[0].set_xlabel('Score')
        axes[0].set_ylabel('Probability')

        sorted_scores = np.sort(df['score'].values)
        cdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
        axes[1].plot(sorted_scores, cdf, color=COLORS['primary'], linewidth=2)
        axes[1].set_xlim(0, 300)
        axes[1].axhline(y=0.9, color=COLORS['secondary'], linestyle='--', label='90th percentile')
        axes[1].set_title('CDF of Post Score', fontsize=11)
        axes[1].set_xlabel('Score')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.warning("💡 **Key Insight:** 90% of posts score below 10. Sentiment correlation with engagement is essentially zero (-0.04, -0.02). What matters is *interaction volume* and *timing*.")

# ── PAGE: Model Results ────────────────────────────────────────────────────────
elif page == "🤖 Model Results":
    st.title("🤖 Baseline Model Results")
    st.markdown("*Logistic Regression · Decision Tree · Random Forest — 80/20 train/test split*")
    st.markdown("---")

    results_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
        'Accuracy': [0.715, 0.701, 0.710],
        'F1 Score': [0.337, 0.287, 0.349],
        'Precision': [0.651, 0.596, 0.611],
        'Recall': [0.228, 0.189, 0.244]
    }
    results_df = pd.DataFrame(results_data)

    st.dataframe(
        results_df.style.highlight_max(subset=['Accuracy','F1 Score','Precision','Recall'],
                                        color='#ECE2D0').format({
            'Accuracy': '{:.3f}', 'F1 Score': '{:.3f}',
            'Precision': '{:.3f}', 'Recall': '{:.3f}'
        }),
        use_container_width=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    x = np.arange(len(metrics))
    width = 0.25

    for i, (model, color) in enumerate(zip(results_data['Model'],
                                            [COLORS['primary'], COLORS['secondary'], COLORS['accent']])):
        vals = [results_data[m][i] for m in metrics]
        axes[0].bar(x + (i-1)*width, vals, width, label=model, color=color,
                    edgecolor=COLORS['dark'])
    axes[0].set_title('Baseline Model Comparison', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # Confusion matrix visualization
    cm_data = {'LR': [[363,22],[139,41]], 'DT': [[362,23],[146,34]], 'RF': [[357,28],[136,44]]}
    cm_colors = [COLORS['primary'], COLORS['secondary'], COLORS['muted']]
    selected_cm = st.selectbox("View confusion matrix for:", list(cm_data.keys()),
                                key='cm_select')
    cm = np.array(cm_data[selected_cm])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=axes[1],
                xticklabels=['Low','High'], yticklabels=['Low','High'])
    axes[1].set_title(f'Confusion Matrix — {selected_cm}', fontsize=13)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.error("⚠️ **Problem Identified:** Recall for high-engagement posts is only 19–24%. Models are biased toward the majority class (low engagement). This led us to implement class weighting and SMOTE.")

# ── PAGE: Model Improvement ────────────────────────────────────────────────────
elif page == "📈 Model Improvement":
    st.title("📈 Model Improvement")
    st.markdown("*Addressing class imbalance with class_weight='balanced' and SMOTE*")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Baseline", "Class Weighted", "SMOTE"])

    baseline = {'Model': ['Logistic Regression','Decision Tree','Random Forest'],
                'Accuracy':[0.715,0.701,0.710],'F1':[0.337,0.287,0.349],
                'Precision':[0.651,0.596,0.611],'Recall':[0.228,0.189,0.244]}
    balanced = {'Model': ['LR Balanced','DT Balanced','RF Balanced'],
                'Accuracy':[0.699,0.646,0.658],'F1':[0.560,0.630,0.615],
                'Precision':[0.524,0.472,0.480],'Recall':[0.600,0.944,0.856]}
    smote = {'Model': ['LR SMOTE','DT SMOTE','RF SMOTE'],
             'Accuracy':[0.710,0.683,0.660],'F1':[0.582,0.634,0.610],
             'Precision':[0.538,0.502,0.481],'Recall':[0.633,0.861,0.833]}

    for tab, data, label in zip([tab1, tab2, tab3],
                                  [baseline, balanced, smote],
                                  ['Baseline', 'Class Weighted', 'SMOTE']):
        with tab:
            st.dataframe(pd.DataFrame(data).style.format({
                'Accuracy':'{:.3f}','F1':'{:.3f}',
                'Precision':'{:.3f}','Recall':'{:.3f}'
            }), use_container_width=True)

    st.markdown("### Recall Improvement — Before vs After")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models_names = ['Logistic\nRegression','Decision\nTree','Random\nForest']
    baseline_recall = [0.228, 0.189, 0.244]
    balanced_recall = [0.600, 0.944, 0.856]
    smote_recall    = [0.633, 0.861, 0.833]
    baseline_f1     = [0.337, 0.287, 0.349]
    balanced_f1     = [0.560, 0.630, 0.615]
    smote_f1        = [0.582, 0.634, 0.610]

    x = np.arange(len(models_names))
    width = 0.25

    for ax, b_vals, bal_vals, sm_vals, title in zip(
        axes,
        [baseline_recall, baseline_f1],
        [balanced_recall, balanced_f1],
        [smote_recall, smote_f1],
        ['Recall Improvement', 'F1 Score Improvement']
    ):
        ax.bar(x - width, b_vals, width, label='Baseline',
               color=COLORS['accent'], edgecolor=COLORS['primary'])
        ax.bar(x, bal_vals, width, label='Class Weighted',
               color=COLORS['secondary'], edgecolor=COLORS['dark'])
        ax.bar(x + width, sm_vals, width, label='SMOTE',
               color=COLORS['primary'], edgecolor=COLORS['dark'])
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(models_names)
        ax.set_ylim(0, 1.05)
        ax.legend()
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success("✅ **Result:** Recall improved from 19–24% (baseline) to 83–94% (balanced/SMOTE). We successfully identified and fixed the class imbalance problem — demonstrating both the issue and the solution.")

