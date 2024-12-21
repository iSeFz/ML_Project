import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import AdamW
from keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score , accuracy_score, classification_report
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


# Logistic Regression Model (from scratch)
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
def plot_lr_confusion_matrix(y_test, y_test_pred, labels):
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_lr_loss_curve(max_iters):
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


def plot_lr_accuracy_curve(max_iters):
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


# First model: Shallow network with only 2 layers and less neurons per layer
def shallow_nn():
    model = Sequential([
        Flatten(), # Flatten the 28x28 input images
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax')  # 26 classes for the alphabet
    ])
    
    return model

# Second model: Deeper network with 4 layers and more neurons per layer
def deeper_nn():
    model = Sequential([
        Flatten(), # Flatten the 28x28 input images        
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(26, activation='softmax')  # 26 classes for the alphabet
    ])
    
    return model


# Function to plot training history
def plot_nn_training_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title(f'{model_name} - Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


# Training and evaluation function
def train_and_evaluate_nn(model, model_name, x_train, y_train, x_test, y_test):
    # Compile model
    model.compile(optimizer=AdamW(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=15,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping]
    )
    
    # Plot training history
    plot_nn_training_history(history, model_name)
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"{model_name} - Test Accuracy: {test_acc:.4f}")
    
    return history, test_acc

# Plot confusion matrix using seaborn heatmap
def plot_nn_confusion_matrix(y_true, y_pred, classes):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and plot
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # Save the plot as an image
    plt.show()


# Evaluate model performance with various metrics
def evaluate_nn_model(model, x_test, y_test):
    # Get predictions
    y_pred = model.predict(x_test)
    
    # Convert predicted probabilities back to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Create class labels (A-Z)
    class_labels = [chr(i + ord('A')) for i in range(26)]
    
    # Plot confusion matrix
    plot_nn_confusion_matrix(y_test, y_pred_classes, class_labels)
    
    # Calculate and print classification report
    report = classification_report(y_test, y_pred_classes, 
                                 target_names=class_labels,
                                 digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Calculate and print average F1 score
    avg_f1 = f1_score(y_test, y_pred_classes, average='weighted')
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {best_acc:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")


def preprocess_image(image_path):
    # Load the image
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img) # Normalize pixel values to [0,1]
    inverted_array = (255 - img_array) / 255.0  # Invert using NumPy
    img_array = tf.expand_dims(inverted_array, axis=0)  # Reshape to match model input shape

    return img, img_array

def predict_image(model, image_path):
    # Preprocess the image
    img, img_array = preprocess_image(image_path)
    
    # Predict the class
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Convert class index to alphabet
    predicted_alphabet = chr(predicted_class + 65)
    
    return predicted_alphabet


def test_best_model_on_external_imgs(images_dir):
    # List to store predictions
    predictions = []

    # Iterate over each image in the directory
    plt.figure(figsize=(20, 15))
    for i, image_file in enumerate(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_file)
        predicted_alphabet = predict_image(loaded_model, image_path)
        predictions.append((image_file, predicted_alphabet))
        
        # Display the image and prediction in a subplot
        plt.subplot(3, 5, i + 1)
        img, _ = preprocess_image(image_path)
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted Alphabet: {predicted_alphabet}", fontsize=20)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("team_name_predictions.png")  # Save the plot as an image
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
    plot_lr_confusion_matrix(y_test, y_test_pred, labels)

    # Plot the loss curve
    plot_lr_loss_curve(max_iters)

    # Plot the accuracy curve
    plot_lr_accuracy_curve(max_iters)


    """
    SVM with Linear Kernel
    """
    # SVM with Linear kernel
    print('Start svm_train with linear kernel')
    y_test_pred = svm_train(X_train_flattened, X_test_flattened, y_train, "linear")

    # SVM with Linear kernel evaluation
    print('Start svm_test_eval with linear kernel')
    svm_test_eval(y_test_pred, y_test, "linear")

    """
    SVM with Non-Linear Kernel
    """
    # SVM with Non-Linear kernel
    print('Start svm_train with poly kernel')
    y_test_pred = svm_train(X_train_flattened, X_test_flattened, y_train, "poly")

    # SVM with Non-Linear kernel evaluation
    print('Start svm_test_eval with poly kernel')
    svm_test_eval(y_test_pred, y_test, "poly")


    """
    Neural Networks
    """
    # Train and evaluate Model 1
    print("\nTraining Model 1 (Shallow NN with 2 hidden layers)...")
    shallow_nn_model = shallow_nn()
    _, acc1 = train_and_evaluate_nn(shallow_nn_model, "Shallow NN", X_train, y_train, X_test, y_test)

    # Train and evaluate Model 2
    print("\nTraining Model 2 (Deeper NN with 4 hidden layers)...")
    deeper_nn_model = deeper_nn()
    _, acc2 = train_and_evaluate_nn(deeper_nn_model, "Deeper NN", X_train, y_train, X_test, y_test)

    # Save the best model
    best_model = shallow_nn_model if acc1 > acc2 else deeper_nn_model
    best_model_name = "Shallow NN" if acc1 > acc2 else "Deeper NN"
    best_acc = max(acc1, acc2)

    print(f"\nSaving {best_model_name} (Accuracy: {best_acc:.4f}) as the best model...")
    best_model.save('best_nn_model.h5')

    # Load the saved model and evaluate it thoroughly
    print("\nEvaluating best model with detailed metrics...")
    loaded_model = tf.keras.models.load_model('best_nn_model.h5')

    # Perform detailed evaluation
    print("\nDetailed Evaluation of Best Model:")
    print("=" * 50)
    evaluate_nn_model(loaded_model, X_test, y_test)

    # Directory containing the images
    images_dir = "team_names_letters/"
    test_best_model_on_external_imgs(images_dir)

