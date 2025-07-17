from flask import Flask, request, jsonify, render_template_string
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
db_path = 'promo_recommender_v2.db'

html_form = """
<!DOCTYPE html>
<html>
<head><title>Promotion Recommender</title></head>
<body>
<h2>Get Personalized Promotions</h2>
<form action="/recommend" method="get">
    <label for="user_id">Enter User ID:</label>
    <input type="number" id="user_id" name="user_id" required>
    <input type="submit" value="Get Recommendations">
</form>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_form)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM purchase_history", conn)
    conn.close()

    # Pivot table for collaborative filtering (user-item matrix)
    user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='purchase_count', fill_value=0)

    if user_id not in user_item_matrix.index:
        return jsonify({'message': 'User not found'}), 404

    # Compute similarity
    user_sim = cosine_similarity(user_item_matrix)
    user_idx = user_item_matrix.index.get_loc(user_id)
    sim_scores = user_sim[user_idx]

    # Weighted sum of other users' preferences
    weighted_scores = np.dot(sim_scores, user_item_matrix.values)
    recommended_ids = np.argsort(weighted_scores)[::-1]

    # Filter out already purchased products
    purchased = user_item_matrix.loc[user_id]
    not_purchased = np.where(purchased.values == 0)[0]
    top_recs = [user_item_matrix.columns[i] for i in recommended_ids if i in not_purchased][:3]

    product_info = df[df['product_id'].isin(top_recs)][['product_id', 'product_name', 'description']].drop_duplicates().to_dict(orient='records')
    return jsonify({'user_id': user_id, 'recommendations': product_info})

if __name__ == '__main__':
    app.run(debug=True)
