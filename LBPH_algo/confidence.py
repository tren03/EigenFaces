import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Load your collected data
distances = np.load('distances.npy')
labels = np.load('labels.npy')

# Reshape distances for sklearn
distances = distances.reshape(-1, 1)

# Train logistic regression model
model = LogisticRegression()
model.fit(distances, labels)

# Save the trained model
joblib.dump(model, 'confidence_model.pkl')
