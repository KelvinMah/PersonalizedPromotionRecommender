# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy app code and database
COPY promo_recommender_collab_app.py /app/
COPY promo_recommender_v2.db /app/

# Install dependencies
RUN pip install flask pandas scikit-learn numpy

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "promo_recommender_collab_app.py"]
