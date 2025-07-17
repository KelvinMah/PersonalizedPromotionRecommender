from flask import Flask, request, jsonify
import sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
db_path = 'promo_recommender.db'

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM purchase_history", conn)
    conn.close()

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['description'])
    product_sim = cosine_similarity(tfidf_matrix)

    user_products = df[df['user_id'] == user_id]['product_id'].values
    user_indices = [df[df['product_id'] == pid].index[0] for pid in user_products if pid in df['product_id'].values]

    if not user_indices:
        return jsonify({'message': 'No products found for this user'}), 404

    sim_scores = product_sim[user_indices].mean(axis=0)
    recommended_indices = np.argsort(sim_scores)[::-1][:3]
    recommended_products = df.iloc[recommended_indices][['product_id', 'description']].to_dict(orient='records')

    return jsonify({'user_id': user_id, 'recommendations': recommended_products})

if __name__ == '__main__':
    app.run(debug=True)
