import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from wordcloud import WordCloud
from gensim.parsing.preprocessing import STOPWORDS

st.set_page_config(
    page_title="Employee Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    import os
    path = os.path.join(os.path.dirname(__file__), 'employee_reviews_final.csv')
    return pd.read_csv(path)

df = load_data()

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")
companies = ['All'] + sorted(df['company'].unique().tolist())
selected_company = st.sidebar.selectbox("Company", companies)
sentiments = ['All', 'positive', 'neutral', 'negative']
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)

filtered = df.copy()
if selected_company != 'All':
    filtered = filtered[filtered['company'] == selected_company]
if selected_sentiment != 'All':
    filtered = filtered[filtered['sentiment_label'] == selected_sentiment]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Employee Survey Sentiment Analyzer")
st.markdown("Analyzing **{:,}** Glassdoor reviews — Amazon, Apple, Facebook, Google, Microsoft, Netflix".format(len(df)))
st.markdown("---")

# ── KPI Cards ────────────────────────────────────────────────────────────────
total = len(filtered)
pct_pos = (filtered['sentiment_label'] == 'positive').sum() / total * 100
pct_neg = (filtered['sentiment_label'] == 'negative').sum() / total * 100
avg_score = filtered['net_score'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{total:,}")
col2.metric("Positive Sentiment", f"{pct_pos:.1f}%")
col3.metric("Negative Sentiment", f"{pct_neg:.1f}%")
col4.metric("Avg Net Score", f"{avg_score:.3f}")

st.markdown("---")

# ── Row 1: Sentiment distribution + Negative by company ──────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sentiment Distribution")
    sent_counts = filtered['sentiment_label'].value_counts()
    colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
    bar_colors = [colors.get(s, '#aaa') for s in sent_counts.index]

    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    bars = ax1.bar(sent_counts.index, sent_counts.values, color=bar_colors, width=0.5)
    for bar, val in zip(bars, sent_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}', ha='center', fontsize=9, fontweight='bold')
    ax1.set_ylabel("Reviews")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

with col_b:
    st.subheader("Negative Sentiment Rate by Company")
    neg_rate = df[df['sentiment_label'] == 'negative'].groupby('company').size() \
               / df.groupby('company').size() * 100
    neg_rate = neg_rate.sort_values(ascending=True)

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.barh(neg_rate.index, neg_rate.values, color='#e74c3c', alpha=0.8, height=0.5)
    for i, val in enumerate(neg_rate.values):
        ax2.text(val + 0.1, i, f'{val:.1f}%', va='center', fontsize=9)
    ax2.set_xlabel("% Negative")
    ax2.set_xlim(0, 12)
    ax2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# ── Row 2: Sentiment by topic ────────────────────────────────────────────────
st.subheader("Sentiment by Employee Concern Topic")

topic_pct = pd.crosstab(
    filtered['topic_name'],
    filtered['sentiment_label'],
    normalize='index'
) * 100

for col in ['positive', 'neutral', 'negative']:
    if col not in topic_pct.columns:
        topic_pct[col] = 0

topic_pct = topic_pct[['positive', 'neutral', 'negative']].sort_values('negative', ascending=True)

fig3, ax3 = plt.subplots(figsize=(10, 3.5))
topic_pct.plot(kind='barh', stacked=True,
               color=['#2ecc71', '#95a5a6', '#e74c3c'], ax=ax3, width=0.6)
ax3.set_xlabel("% of Reviews")
ax3.set_ylabel("")
ax3.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax3.legend(title='Sentiment', bbox_to_anchor=(1.01, 1), loc='upper left')
ax3.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.markdown("---")

# ── Row 3: Word cloud ─────────────────────────────────────────────────────────
st.subheader("Most Common Words in Negative Reviews")

neg_text = ' '.join(
    filtered[filtered['sentiment_label'] == 'negative']['cons'].dropna().values
)

if len(neg_text.strip()) > 50:
    wc = WordCloud(width=900, height=350, background_color='white',
                   colormap='Reds', max_words=80,
                   stopwords=set(STOPWORDS), collocations=False).generate(neg_text)
    fig4, ax4 = plt.subplots(figsize=(11, 4))
    ax4.imshow(wc, interpolation='bilinear')
    ax4.axis('off')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()
else:
    st.info("Not enough negative reviews to generate word cloud with current filters.")

st.markdown("---")

# ── Row 4: Review Explorer ────────────────────────────────────────────────────
st.subheader("📝 Review Explorer")
st.markdown("Browse individual reviews based on your filters above.")

sample = filtered[['company', 'rating', 'sentiment_label',
                    'net_score', 'pros', 'cons']].copy()
sample = sample.rename(columns={
    'company': 'Company', 'rating': 'Rating',
    'sentiment_label': 'Sentiment', 'net_score': 'Net Score',
    'pros': 'Pros', 'cons': 'Cons'
})
sample['Net Score'] = sample['Net Score'].round(3)
st.dataframe(sample.head(50), use_container_width=True)

st.markdown("---")
st.caption("Project 2 — Employee Sentiment Analyzer | People Analytics Portfolio | Aakash Malhan")
