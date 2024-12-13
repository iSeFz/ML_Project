import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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


