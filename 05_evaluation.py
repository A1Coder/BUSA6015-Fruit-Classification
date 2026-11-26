# BUSA6015 - Deep Learning
# Fruit Quality Classification
# Stage 6: Final held-out test evaluation (no further training)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'outputs'

# 1. Freeze model selection using validation loss, before test evaluation.
validation = pd.concat([
    pd.read_csv(OUTPUT_DIR / 'baseline_results.csv'),
    pd.read_csv(OUTPUT_DIR / 'regularised_results.csv')
], ignore_index=True)
selected_name = validation.loc[validation['validation_loss'].idxmin(), 'model']
print('Selected CNN (minimum validation loss):', selected_name)
print(validation[['model', 'best_epoch', 'validation_accuracy',
                  'validation_loss', 'validation_macro_f1']].to_string(index=False))

# 2. Load the unchanged test set. Training labels are used only to define
# the majority-class benchmark. No model fitting occurs in this script.
with np.load(OUTPUT_DIR / 'processed_data.npz') as data:
    X_test = data['X_test']
    y_test = data['y_test']
    y_train = data['y_train']
    class_names = data['class_names'].astype(str).tolist()

labels = np.arange(len(class_names))
print('\nTest images:', len(y_test))
majority_label = int(np.bincount(y_train).argmax())
print('Majority training class:', class_names[majority_label])

# 3. Common evaluation metrics for all methods.
results = []
reports = []
matrices = {}

def evaluate_predictions(name, y_pred, loss=np.nan):
    results.append({
        'model': name,
        'selected': name == selected_name,
        'test_images': len(y_test),
        'correct_predictions': int(np.sum(y_pred == y_test)),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_macro_precision': precision_score(y_test, y_pred, labels=labels,
                                                 average='macro', zero_division=0),
        'test_macro_recall': recall_score(y_test, y_pred, labels=labels,
                                          average='macro', zero_division=0),
        'test_macro_f1': f1_score(y_test, y_pred, labels=labels,
                                  average='macro', zero_division=0),
        'test_loss': loss
    })
    report = classification_report(y_test, y_pred, labels=labels,
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    for cls in class_names:
        reports.append({'model': name, 'class': cls, **report[cls]})
    matrices[name] = confusion_matrix(y_test, y_pred, labels=labels)

# Benchmark: always predict the most frequent training class.
y_majority = np.full(y_test.shape, majority_label, dtype=y_test.dtype)
evaluate_predictions('Majority-class benchmark', y_majority)

# 4. Evaluate the two saved minimum-validation-loss CNN checkpoints.
for name, filename in [
    ('Baseline CNN', 'baseline_cnn.keras'),
    ('Regularised CNN', 'regularised_cnn.keras')
]:
    model = models.load_model(OUTPUT_DIR / filename)
    metrics = model.evaluate(X_test, y_test, batch_size=32,
                             verbose=0, return_dict=True)
    probabilities = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    evaluate_predictions(name, y_pred, loss=float(metrics['loss']))

# 5. Save and display test results.
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / 'test_results.csv', index=False)
pd.DataFrame(reports).to_csv(OUTPUT_DIR / 'test_classification_report.csv', index=False)
print('\n===== FINAL TEST RESULTS =====')
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\n===== SELECTED CNN: PER-CLASS TEST RESULTS =====')
print(pd.DataFrame(reports).query('model == @selected_name').to_string(
    index=False, float_format=lambda x: f'{x:.4f}'))

# 6. Benchmark comparison chart. All metrics are on the same test set.
metric_cols = ['test_accuracy', 'test_macro_precision',
               'test_macro_recall', 'test_macro_f1']
plot_data = results_df.set_index('model')[metric_cols].mul(100).T
plot_data.index = ['Accuracy', 'Macro precision', 'Macro recall', 'Macro F1']
ax = plot_data.plot.bar(figsize=(10, 5), rot=0, width=0.78)
ax.set_ylim(0, 110)
ax.set_ylabel('Score (%)')
ax.set_title('Held-out Test Performance by Model')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
ax.grid(axis='y', alpha=0.25)
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
ax.figure.tight_layout()
ax.figure.savefig(OUTPUT_DIR / 'test_benchmark_comparison.png', dpi=300,
                  bbox_inches='tight')

# 7. Selected CNN confusion matrix; raw counts, actual rows, predicted columns.
pretty_names = ['Fresh apple', 'Fresh banana', 'Fresh orange',
                'Rotten apple', 'Rotten banana', 'Rotten orange']
name_map = dict(zip(['freshapples', 'freshbanana', 'freshoranges',
                     'rottenapples', 'rottenbanana', 'rottenoranges'], pretty_names))
axis_names = [name_map.get(cls, cls) for cls in class_names]
cm = matrices[selected_name]
pd.DataFrame(cm, index=class_names, columns=class_names).rename_axis(
    'actual').to_csv(OUTPUT_DIR / 'test_confusion_matrix.csv')
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=axis_names, yticklabels=axis_names, ax=ax)
ax.set_xlabel('Predicted class')
ax.set_ylabel('Actual class')
ax.set_title(f'{selected_name}: Test Confusion Matrix')
ax.tick_params(axis='x', rotation=35)
ax.tick_params(axis='y', rotation=0)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'test_confusion_matrix.png', dpi=300)

# 8. Business-relevant error: rotten fruit predicted as any fresh class.
fresh = [i for i, cls in enumerate(class_names) if cls.startswith('fresh')]
rotten = [i for i, cls in enumerate(class_names) if cls.startswith('rotten')]
rotten_count = int(cm[rotten, :].sum())
missed_rotten = int(cm[np.ix_(rotten, fresh)].sum())
print(f'\nSelected CNN: rotten images predicted fresh: {missed_rotten}/{rotten_count}')
if rotten_count:
    print(f'Rotten-to-fresh error rate: {missed_rotten / rotten_count:.2%}')
print('\nEvaluation completed. Outputs saved to:', OUTPUT_DIR)
plt.show()
