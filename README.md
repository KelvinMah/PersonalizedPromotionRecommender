# 🛍️ Personalized Promotion Recommender

A lightweight, containerized Flask application that recommends promotional products to users based on their purchase history using both **collaborative filtering** and **content-based similarity** (TF-IDF on product descriptions).

---

## 📦 Features

- 🔄 **Collaborative Filtering** using purchase frequency (implicit feedback)
- 🧠 **Content-Based Recommendations** using TF-IDF embeddings on product descriptions
- 📊 SQLite-backed product and user purchase database
- 📥 Simple web form for user interaction
- 🐳 Dockerized for easy deployment
- 📈 JSON API for integration with other services

---

## 🏁 Getting Started

### 🔧 Requirements

- Docker installed
- (Optional) Python 3.10+ if running locally without Docker

---

### 🚀 Run with Docker

```bash
# 1. Clone this repo and navigate into it
git clone https://github.com/yourusername/promo-recommender.git
cd promo-recommender

# 2. Build the Docker image
docker build -t promo-recommender .

# 3. Run the container
docker run -p 5000:5000 promo-recommender
