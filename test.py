"""
COMP 472 Artificial Intelligence Summer 2025
Mini Project 1

Team Members:
- Johnny Dang 40245598
- Julia Trinh 40245980
- Oviya Sinnathamby 40249479
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("="*60)
print("MNIST Handwritten Digit Classification")
print("="*60)

# 1. DATA LOADING AND EXPLORATION
print("\n1. Loading MNIST Dataset...")
print("-" * 30)

# Fetch the MNIST dataset (70,000 samples of 28x28 grayscale images)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype('float32')  # Feature data (pixel values)
y = mnist.target.astype('int')    # Target labels (digits 0-9)

print(f"Dataset shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Data type - Features: {X.dtype}, Labels: {y.dtype}")
print(f"Pixel value range: {X.min()} to {X.max()}")
print(f"Unique labels: {np.unique(y)}")

# 2. DATA PREPROCESSING WITH NUMPY
print("\n2. Data Preprocessing with NumPy...")
print("-" * 30)

# Normalize pixel values to range [0,1] using NumPy operations
X_normalized = X / 255.0
print(f"Normalized pixel value range: {X_normalized.min()} to {X_normalized.max()}")

# Display dataset statistics using NumPy
print(f"Mean pixel value: {np.mean(X_normalized):.4f}")
print(f"Standard deviation: {np.std(X_normalized):.4f}")

# Count samples per digit class
unique_labels, counts = np.unique(y, return_counts=True)
print("Class distribution:")
for label, count in zip(unique_labels, counts):
    print(f"  Digit {label}: {count} samples")

# 3. DATA VISUALIZATION
print("\n3. Visualizing Sample Images...")
print("-" * 30)

def plot_sample_images(images, labels, title="Sample Images", n_samples=20):
    """
    Plot a grid of sample images with their corresponding labels
    
    Args:
        images: Array of image data
        labels: Array of corresponding labels
        title: Title for the plot
        n_samples: Number of samples to display
    """
    # Create a random selection of samples
    indices = np.random.choice(len(images), n_samples, replace=False)
    sample_images = images[indices]
    sample_labels = labels[indices]
    
    # Calculate grid dimensions
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division
    
    # Create the plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)
    
    for i in range(n_samples):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes
        else:
            ax = axes[row, col]
        
        # Reshape 784-dimensional vector back to 28x28 image
        image = sample_images[i].reshape(28, 28)
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Label: {sample_labels[i]}')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes
        else:
            ax = axes[row, col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Display sample images from the dataset
plot_sample_images(X_normalized, y, "Original MNIST Dataset Samples")

# 4. DATA SPLITTING
print("\n4. Splitting Data (80% Training, 20% Testing)...")
print("-" * 30)

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducible results
    stratify=y          # Maintain class distribution in both sets
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training set size: {len(y_train)} samples ({len(y_train)/len(y)*100:.1f}%)")
print(f"Testing set size: {len(y_test)} samples ({len(y_test)/len(y)*100:.1f}%)")

# Verify class distribution is maintained
train_unique, train_counts = np.unique(y_train, return_counts=True)
test_unique, test_counts = np.unique(y_test, return_counts=True)
print("\nClass distribution after split:")
print("Digit | Train | Test")
print("-" * 20)
for i in range(10):
    print(f"  {i}   | {train_counts[i]:5d} | {test_counts[i]:4d}")

# 5. MODEL TRAINING
print("\n5. Training Logistic Regression Model...")
print("-" * 30)

# What is Logistic Regression?
print("About Logistic Regression:")
print("- A statistical method used for binary and multiclass classification")
print("- Uses the logistic function to model probability of class membership")
print("- For multiclass (like our 10 digits), it uses 'one-vs-rest' approach")
print("- Finds optimal weights through maximum likelihood estimation")
print("- Simple, fast, and interpretable baseline classifier")

# Create and train the Logistic Regression model
# Using limited iterations and tolerance for faster training on large dataset
classifier = LogisticRegression(
    max_iter=1000,      # Maximum number of iterations
    random_state=42,    # For reproducible results
    solver='lbfgs'      # Efficient solver for multiclass problems
)

print("\nTraining the model...")
classifier.fit(X_train, y_train)
print("Training completed!")

# 6. MODEL EVALUATION
print("\n6. Evaluating Model Performance...")
print("-" * 30)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
train_accuracy = classifier.score(X_train, y_train)
test_accuracy = classifier.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Classification Report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print("The classification report shows precision, recall, and F1-score for each digit:")
print("- Precision: Of all predicted class X, how many were actually class X?")
print("- Recall: Of all actual class X, how many were correctly predicted?")
print("- F1-score: Harmonic mean of precision and recall")
print("- Support: Number of actual samples for each class")
print()
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
print("Shows actual vs predicted classifications:")
print("- Rows represent actual digits")
print("- Columns represent predicted digits")
print("- Diagonal values show correct predictions")
print("- Off-diagonal values show misclassifications")
print()

# Calculate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (Actual vs Predicted):")
print("   ", end="")
for i in range(10):
    print(f"{i:4d}", end="")
print()

for i in range(10):
    print(f"{i}: ", end="")
    for j in range(10):
        print(f"{cm[i,j]:4d}", end="")
    print()

# Visualize confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - MNIST Digit Classification')
plt.xlabel('Predicted Digit')
plt.ylabel('Actual Digit')
plt.show()

# 7. PREDICTION VISUALIZATION
print("\n7. Visualizing Predictions...")
print("-" * 30)

# Show some test predictions
plot_sample_images(X_test, y_pred, "Model Predictions on Test Set")

# Show some misclassified examples
misclassified_indices = np.where(y_test != y_pred)[0]
if len(misclassified_indices) > 0:
    print(f"\nFound {len(misclassified_indices)} misclassified samples")
    
    # Plot some misclassified examples
    n_misclassified = min(20, len(misclassified_indices))
    mis_indices = np.random.choice(misclassified_indices, n_misclassified, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle('Misclassified Examples (Actual → Predicted)', fontsize=16)
    
    for i in range(n_misclassified):
        row = i // 5
        col = i % 5
        idx = mis_indices[i]
        
        image = X_test[idx].reshape(28, 28)
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'{y_test[idx]} → {y_pred[idx]}', color='red')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# 8. SUMMARY
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"✓ Successfully loaded MNIST dataset with {len(X)} samples")
print(f"✓ Used NumPy for data normalization and manipulation")
print(f"✓ Split data into 80% training ({len(y_train)} samples) and 20% testing ({len(y_test)} samples)")
print(f"✓ Trained Logistic Regression classifier")
print(f"✓ Achieved {test_accuracy*100:.2f}% accuracy on test set")
print(f"✓ Generated classification report and confusion matrix")
print(f"✓ Visualized sample images and predictions")
print("\nThe classifier successfully recognizes handwritten digits with good accuracy!")
print("="*60)