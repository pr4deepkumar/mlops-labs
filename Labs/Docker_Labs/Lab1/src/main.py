# Import necessary libraries
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

if __name__ == '__main__':
    # Load the Digits dataset (1,797 8x8 images of handwritten digits 0-9)
    digits = load_digits()
    X, y = digits.data, digits.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a Support Vector Machine classifier
    model = SVC(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train, y_train)

    # Check the model on the test set it has never seen
    print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")

    # Save the model to the model/ folder (mount it with -v to keep the file after the container exits)
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/digits_model.pkl')
    
    print("The model training was successful")
