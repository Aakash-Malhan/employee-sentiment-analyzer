# Employee Survey Sentiment Analyzer
**People Analytics Portfolio — Project 2 | Aakash Malhan**

A natural language processing dashboard that analyzes 67,529 real Glassdoor employee reviews across Amazon, Apple, Facebook, Google, Microsoft, and Netflix using VADER sentiment scoring and LDA topic modeling.

🔗 **[Live App on Hugging Face Spaces]([https://huggingface.co/spaces/aakash-malhan/employee-sentiment-analyzer](https://huggingface.co/spaces/aakash-malhan/employee-sentiment-analyzer))**

---

## Project Overview

HR and People Analytics teams rely on employee survey data to identify systemic issues around culture, management, workload, and growth. This project simulates that workflow end-to-end — from raw text data to an interactive executive dashboard.

This project is part of a Microsoft HR Data Analyst (HRBI) portfolio demonstrating applied NLP, people analytics thinking, and business communication of insights.

---

## Key Findings

- **86% of reviews score positive** — driven by the structural nature of Glassdoor's pros/cons format where positive language in pros dominates combined scores
- **Amazon has the highest negative sentiment rate (7.4%)**, Facebook the lowest (4.0%); Microsoft sits at 5.0%
- **Customer-Facing Work** is the most negatively charged topic (11.8% negative) — concentrated in Amazon retail reviews
- **Management & Work Culture** is the dominant complaint theme, accounting for 93% of all reviews by topic volume
- **Workload & Burnout** language appears even in positive reviews — indicating it is a near-universal experience rather than a purely negative one

---

<img width="690" height="390" alt="download" src="https://github.com/user-attachments/assets/48aa753c-fc11-42ef-b157-9b6babf05f66" />
<img width="790" height="390" alt="download (1)" src="https://github.com/user-attachments/assets/8b895db9-ab36-4861-8a0e-923f52d2ba9a" />
<img width="888" height="490" alt="download (2)" src="https://github.com/user-attachments/assets/ec6c9568-653e-4db5-b072-10b2d864b27c" />


## Tools & Methods

| Layer | Tool |
|---|---|
| Data source | Glassdoor employee reviews (Kaggle, 67,529 rows) |
| Environment | Google Colab |
| Sentiment scoring | VADER (vaderSentiment) |
| Topic modeling | LDA — Latent Dirichlet Allocation (Gensim) |
| Visualization | Matplotlib, WordCloud |
| Dashboard | Streamlit |
| Hosting | Hugging Face Spaces |

---

## Methodology

### Sentiment Scoring
VADER was applied separately to the `pros` and `cons` columns rather than combined review text. A **net score** (pros compound score minus cons compound score) was used to assign final sentiment labels. This approach outperformed combined-text scoring, doubling the negative detection rate for 1-2 star reviews.

### Topic Modeling
LDA was trained on the `cons` column only, using a vocabulary of 4,732 words across 67,529 documents. Five topics were discovered:

| Topic | Key Words | Interpretation |
|---|---|---|
| 1 | politics, team, career, slow, growth | Career Growth & Politics |
| 2 | work, management, people, managers, balance | Management & Work Culture |
| 3 | like, customers, retail, want | Customer-Facing Work |
| 4 | microsoft, process, performance, years | Corporate Process & Performance |
| 5 | time, hours, long, pressure, working | Workload & Burnout |

### Validation
VADER sentiment labels were cross-tabulated against star ratings to assess model accuracy. The net-score approach showed a clear directional relationship: 1-2 star reviews had 2.6x the negative rate of 4-5 star reviews.

---

## Dashboard Features

- Filter by company and sentiment
- KPI cards: total reviews, % positive, % negative, avg net score
- Sentiment distribution bar chart
- Negative sentiment rate by company (horizontal bar)
- Sentiment breakdown by LDA topic (stacked bar)
- Word cloud of most common words in negative reviews
- Interactive review explorer (browse raw pros/cons)

---

## Repository Structure
