import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Load the datasets
df_train = pd.read_csv('mitbih_train.csv')
df_test = pd.read_csv('mitbih_test.csv')

# the data doesn't have names for the columns, so we add them
# From column 0 to 187 are the values of the ECG signal
# The last column is the label of the signal
df_train.columns = [str(i) for i in range(188)]
df_train = df_train.rename(columns={df_train.columns[-1]: 'target'})
df_test.columns = [str(i) for i in range(188)]
df_test = df_test.rename(columns={df_test.columns[-1]: 'target'})

# Function to balance the dataset
def balance_dataset(df, target_column, n_samples):
    classes = df[target_column].unique()
    balanced_dfs = []

    for cls in classes:
        df_class = df[df[target_column] == cls]

        if len(df_class) > n_samples:
            # Undersample the majority class
            df_class_balanced = df_class.sample(n=n_samples, random_state=42)
        else:
            # Oversample the minority class
            df_class_balanced = resample(df_class, replace=True, n_samples=n_samples, random_state=42)

        balanced_dfs.append(df_class_balanced)

    # Combine and shuffle 
    df_balanced = pd.concat(balanced_dfs)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

# Balance the training and test datasets
df_train_balanced = balance_dataset(df_train, 'target', 4000)
df_test_balanced = balance_dataset(df_test, 'target', 162)

# Prepare the data
X_train = df_train_balanced.drop('target', axis=1).values
y_train = df_train_balanced['target'].values
X_test = df_test_balanced.drop('target', axis=1).values
y_test = df_test_balanced['target'].values

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the LSTM model
model = Sequential()

model.add(LSTM(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))  # 5 classes

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Other metrics
f1 = f1_score(y_true, y_pred_classes, average='weighted')
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')

print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# ROC Curve
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('(ROC) Curve')
plt.legend(loc="lower right")
plt.show()