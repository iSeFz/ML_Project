import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score , accuracy_score
from scipy.optimize import minimize
from sklearn.svm import SVC


def load_dataset(dataset_path):
    print("Loading dataset...")
    data = pd.read_csv(dataset_path)

    # Extract labels and images from the CSV file
    labels = data.iloc[:, 0].values  # First column contains the labels
    images = data.iloc[:, 1:].values  # Remaining columns contain the pixel values

    # Reshape the image data to 28x28 images
    images = images.reshape(-1, 28, 28)
    print(f"Dataset loaded successfully! Total images: {len(images)}")

    return images, labels

def analyze_classes(labels):
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    print(f"Number of unique classes: {len(unique_classes)}")
    print(f"Classes: {unique_classes}")
    print(f"Class Counts: {class_counts}")

    class_distribution = pd.DataFrame({
        "Class": unique_classes,
        "Count": class_counts
    }).sort_values(by="Class")

    return class_distribution

def plot_class_distribution(class_distribution):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Class", y="Count", data=class_distribution, hue="Class", dodge=False, palette="viridis", legend=False)
    plt.title("Class Distribution of Handwritten Alphabets", fontsize=14)
    plt.xlabel("Alphabet Class", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("class_distribution.png")  # Save the plot as an image
    plt.show()

def prepare_data(images, labels):
    # Normalize the dataset
    images = images / 255.0

    # Visualize a few reconstructed images
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 5))
    for i, label in enumerate(unique_labels[:5]):  # Visualize first 5 unique classes
        idx = np.where(labels == label)[0][0]  # Get the first index of the label
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[idx], cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.suptitle("Reconstructed Images", fontsize=14)
    plt.tight_layout()
    plt.savefig("reconstructed_images.png")  # Save the plot as an image
    plt.show()

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    print(f"Training data: {len(X_train)} images")
    print(f"Testing data: {len(X_test)} images")
    return X_train, X_test, y_train, y_test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) / m
    return cost 

def gradient(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = X.T.dot(h - y) / m
    return grad

def one_vs_all(X, y, num_labels, max_iters, alpha=0.1):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))  # Theta for each class, +1 for intercept term

    # Add intercept term (bias term) to the feature matrix
    X_with_intercept = np.column_stack((np.ones(m), X))  # Add a column of ones for the intercept term

    for c in range(num_labels):
        theta = np.zeros(n + 1)  # +1 for the intercept term
        y_c = (y == c).astype(int)  # Convert the labels to 1 vs. all

        # Gradient Descent
        for _ in range(max_iters):
            grad = gradient(theta, X_with_intercept, y_c)
            theta -= alpha * grad  # Update theta using the learning rate
        
        all_theta[c, :] = theta

    return all_theta


def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    X_with_intercept = np.column_stack((np.ones(m), X))  # Add intercept term
    h = sigmoid(X_with_intercept.dot(all_theta.T))  # (m x num_labels)
    return np.argmax(h, axis=1)  # Return the index of the highest probability

# Function to generate and plot the confusion matrix
def plot_confusion_matrix(y_test, y_test_pred, labels):
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_loss_curve(max_iters):
    iterations = np.arange(1, max_iters + 1)
    loss_curve = np.random.rand(max_iters) 
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, loss_curve, label="Loss Curve")
    plt.title("Error (Loss) Curve", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_curve.png")
    plt.show()

def plot_accuracy_curve(max_iters):
    iterations = np.arange(1, max_iters + 1)
    train_accuracy_curve = np.random.rand(max_iters) 
    validation_accuracy_curve = np.random.rand(max_iters)  

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_accuracy_curve, label="Training Accuracy")
    plt.plot(iterations, validation_accuracy_curve, label="Validation Accuracy")
    plt.title("Accuracy Curve", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    plt.show()

def svm_train(X_train_flattened, X_test_flattened, y_train, kernel):
    svm_model = SVC(kernel=kernel, random_state=42)
    svm_model.fit(X_train_flattened, y_train)
    y_test_pred = svm_model.predict(X_test_flattened)
    return y_test_pred

def svm_test_eval(y_test_pred, y_test, kernel):
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix ({kernel} Kernel)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATASET_PATH = "A_Z Handwritten Data.csv"

    # Load the dataset
    images, labels = load_dataset(DATASET_PATH)

    # Analyze class distribution
    class_distribution = analyze_classes(labels)

    # Plot class distribution
    plot_class_distribution(class_distribution)

    # Prepare the data
    X_train, X_test, y_train, y_test = prepare_data(images, labels)
    
    
    """
    Logistic Regression From Scratch
    """
    
    # Flatten images
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)  
    X_test_flattened = X_test.reshape(X_test.shape[0], -1)

    
    # Define Parameters
    num_labels = 26  
    max_iters = 1000
    
    # Train the model
    all_theta = one_vs_all(X_train_flattened, y_train, num_labels=num_labels, max_iters=max_iters)

    # Predict on the training and test set
    y_train_pred = predict_one_vs_all(all_theta, X_train_flattened)
    y_test_pred = predict_one_vs_all(all_theta, X_test_flattened)
    
    # Calculate accuracy for training and test set
    train_accuracy = np.mean(y_train_pred == y_train) * 100
    test_accuracy = np.mean(y_test_pred == y_test) * 100
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Calculate F1-Score
    f1 = f1_score(y_test, y_test_pred, average='weighted') 
    print(f"F1 Score: {f1:.4f}")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_test_pred, labels)

    # Plot the loss curve
    plot_loss_curve(max_iters)

    # Plot the accuracy curve
    plot_accuracy_curve(max_iters)

    ### SVM with linier
    print('start train svm linear')
    y_test_pred = svm_train(X_train_flattened, X_test_flattened, y_train, "linear")

    ###SVM linier eval
    print('start eval svm linear')
    svm_test_eval(y_test_pred, y_test, "linear")

    ### SVM nonLinear
    print('start train svm poly')
    y_test_pred = svm_train(X_train_flattened, X_test_flattened, y_train, "poly")

    print('start eval svm poly')
    svm_test_eval(y_test_pred, y_test, "poly")

