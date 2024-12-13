import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(dataset_path):
    print("Loading dataset...")
    data = pd.read_csv(dataset_path)

    # Extract labels and images from the CSV file
    labels = data.iloc[:, 0].values  # First column contains the labels
    images = data.iloc[:, 1:].values  # Remaining columns contain the pixel values

    # Reshape the image data to 28x28 and normalize pixel values to [0, 1]
    images = images.reshape(-1, 28, 28) / 255.0
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

if __name__ == "__main__":
    DATASET_PATH = "A_Z Handwritten Data.csv"

    # Load the dataset
    images, labels = load_dataset(DATASET_PATH)

    # Analyze class distribution
    class_distribution = analyze_classes(labels)

    # Plot class distribution
    plot_class_distribution(class_distribution)
