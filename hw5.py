from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#Part 2
print("Part 2")
data = load_breast_cancer()
num_samples, num_features = data.data.shape

print('-' * 30)
print(f"Number of samples: {num_samples}")
print(f"Number of input features: {num_features}")
print("Class counts:")
for class_id, class_name in enumerate(data.target_names):
    count = (data.target == class_id).sum()
    print(f"  {class_name}: {count}")
print('-' * 30)

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, 
        test_size=0.2, random_state=42, stratify=data.target)
print()
print('-' * 30)
print("Raw data")
print('-' * 30)

#decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Decision Tree Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#random forest classifier
random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Random Forest Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(f"kNN Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"kNN Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print(f"Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Gaussian Naive Bayes Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#SVM classifier
svm = SVC(kernel='rbf', random_state=42) #radial basis function 高斯核
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"SVM Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#Part 3
# Save Part 2 scores before the classifier variables are replaced.
raw_performance = {}
for model_name, classifier in [
    ('Decision Tree', decision_tree), ('Random Forest', random_forest),
    ('kNN', knn), ('Gaussian NB', gnb), ('SVM', svm)
]:
    predictions = classifier.predict(x_test)
    raw_performance[model_name] = (
        accuracy_score(y_test, predictions),
        f1_score(y_test, predictions, average='macro')
    )

#standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
std_x_train = scaler.fit_transform(x_train)
std_x_test = scaler.transform(x_test)

print()
print('-' * 30)
print("Part 3: After standardization")
print('-' * 30)

#decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(std_x_train, y_train)
y_pred = decision_tree.predict(std_x_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Decision Tree Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#random forest classifier
random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
random_forest.fit(std_x_train, y_train)
y_pred = random_forest.predict(std_x_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Random Forest Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(std_x_train, y_train)
y_pred = knn.predict(std_x_test)
print(f"kNN Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"kNN Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(std_x_train, y_train)
y_pred = gnb.predict(std_x_test)
print(f"Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Gaussian Naive Bayes Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

#SVM classifier
svm = SVC(kernel='rbf', random_state=42) #radial basis function 高斯核
svm.fit(std_x_train, y_train)
y_pred = svm.predict(std_x_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"SVM Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

# Compare performance before and after standardization on the same test set.
from pathlib import Path
import matplotlib.pyplot as plt

standardized_performance = {}
for model_name, classifier in [
    ('Decision Tree', decision_tree), ('Random Forest', random_forest),
    ('kNN', knn), ('Gaussian NB', gnb), ('SVM', svm)
]:
    predictions = classifier.predict(std_x_test)
    standardized_performance[model_name] = (
        accuracy_score(y_test, predictions),
        f1_score(y_test, predictions, average='macro')
    )

model_names = list(raw_performance)
positions = list(range(len(model_names)))
bar_width = 0.36
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for metric_index, (ax, metric_name) in enumerate(zip(axes, ['Accuracy', 'Macro-F1'])):
    before_bars = ax.bar(
        [x - bar_width / 2 for x in positions],
        [raw_performance[name][metric_index] for name in model_names],
        width=bar_width, label='Before standardization', color='tab:blue'
    )
    after_bars = ax.bar(
        [x + bar_width / 2 for x in positions],
        [standardized_performance[name][metric_index] for name in model_names],
        width=bar_width, label='After standardization', color='tab:orange'
    )
    ax.bar_label(before_bars, fmt='%.4f', padding=3, fontsize=8)
    ax.bar_label(after_bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set_xticks(positions, model_names, rotation=15)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(metric_name)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right')
axes[0].set_ylabel('Test score')
fig.suptitle('Breast cancer: performance before and after standardization')
fig.tight_layout()
standardization_plot_path = Path(__file__).resolve().parent / 'standardization_performance_comparison.png'
fig.savefig(standardization_plot_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nStandardization comparison plot saved to: {standardization_plot_path}')

#Part 4
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone
import csv
import json

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print()
print('-' * 30)
print("Part 4")
print('-' * 30)

# All candidates use the same five folds of the original 80% training data.
# Scaling is fitted inside each fold; Decision Tree uses raw features.
plot_experiments = [
    ("kNN", "n_neighbors", [1, 3, 5, 9, 15], KNeighborsClassifier()),
    ("Decision Tree", "max_depth", [2, 3, 5, 10, None],
     DecisionTreeClassifier(random_state=42)),
    ("SVM (RBF)", "C", [0.1, 1, 10], SVC(kernel='rbf', random_state=42)),
]

cv_results = {}
selected_parameters = {}
selected_models = {}
cv_rows = []
cv_folds = list(kf.split(x_train, y_train))
for model_name, parameter, values, estimator in plot_experiments:
    candidates = []
    for value in values:
        candidate = clone(estimator).set_params(**{parameter: value})
        if model_name != 'Decision Tree':
            candidate = make_pipeline(StandardScaler(), candidate)
        scores = cross_validate(
            candidate, x_train, y_train, cv=cv_folds,
            scoring={'accuracy': 'accuracy', 'macro_f1': 'f1_macro'},
            error_score='raise'
        )
        result = {
            'value': value,
            'accuracy_mean': float(scores['test_accuracy'].mean()),
            'accuracy_std': float(scores['test_accuracy'].std()),
            'macro_f1_mean': float(scores['test_macro_f1'].mean()),
            'macro_f1_std': float(scores['test_macro_f1'].std()),
            'fold_accuracy': scores['test_accuracy'].tolist(),
            'fold_macro_f1': scores['test_macro_f1'].tolist(),
        }
        candidates.append(result)
        cv_rows.append({'model': model_name, 'parameter': parameter,
                        **{key: val for key, val in result.items() if not key.startswith('fold_')}})
        print(f"{model_name}: {parameter}={value}, "
              f"CV Macro-F1={result['macro_f1_mean']:.4f}, "
              f"Macro-F1 std={result['macro_f1_std']:.4f}")
    # Predefined rule: highest mean Macro-F1, then lower std for an exact tie.
    # If both tie, retain the first candidate in the listed order.
    best = max(candidates, key=lambda row: (row['macro_f1_mean'], -row['macro_f1_std']))
    selected_parameters[model_name] = {parameter: best['value']}
    selected = clone(estimator).set_params(**{parameter: best['value']})
    if model_name != 'Decision Tree':
        selected = make_pipeline(StandardScaler(), selected)
    selected_models[model_name] = selected
    cv_results[model_name] = candidates
    print(f"Selected {model_name}: {parameter}={best['value']}")

# Plot cached CV results so the figure and parameter selection agree exactly.
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
std_axes = []
max_macro_f1_std = 0.0
for ax, (model_name, parameter, values, estimator) in zip(axes, plot_experiments):
    accuracy_means = [row['accuracy_mean'] for row in cv_results[model_name]]
    accuracy_stds = [row['accuracy_std'] for row in cv_results[model_name]]
    macro_f1_means = [row['macro_f1_mean'] for row in cv_results[model_name]]
    macro_f1_stds = [row['macro_f1_std'] for row in cv_results[model_name]]

    # Treat tested settings as categories so that max_depth=None has a label.
    positions = list(range(len(values)))
    ax.errorbar(positions, accuracy_means, yerr=accuracy_stds,
                fmt='o-', capsize=4, label='Accuracy', color='tab:blue')
    ax.errorbar(positions, macro_f1_means, yerr=macro_f1_stds,
                fmt='s--', capsize=4, label='Macro-F1', color='tab:orange')

    # Show Macro-F1 variability separately; error bars above remain mean +/- std.
    std_ax = ax.twinx()
    std_ax.plot(positions, macro_f1_stds, 'D:', color='tab:green',
                linewidth=2, markersize=5, label='Macro-F1 std (right axis)')
    std_ax.tick_params(axis='y', colors='tab:green')
    std_ax.spines['right'].set_color('tab:green')
    std_axes.append(std_ax)
    max_macro_f1_std = max(max_macro_f1_std, max(macro_f1_stds))

    ax.set_xticks(positions, [str(value) for value in values])
    ax.set_xlabel(parameter)
    ax.set_title(model_name)
    ax.grid(axis='y', alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    std_handles, std_labels = std_ax.get_legend_handles_labels()
    ax.legend(handles + std_handles, labels + std_labels,
              loc='lower left', fontsize=8)

axes[0].set_ylabel('CV score (mean +/- 1 standard deviation)')
# Use the same zero-based std scale in all three panels for fair comparison.
for std_ax in std_axes:
    std_ax.set_ylim(0, max(max_macro_f1_std * 1.2, 0.01))
std_axes[-1].set_ylabel('Macro-F1 standard deviation', color='tab:green')
fig.suptitle('Hyperparameter vs. CV performance | 5-fold stratified CV on training data')
fig.tight_layout()
plot_path = Path(__file__).resolve().parent / 'hyperparameter_vs_cv_performance.png'
fig.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nCV performance plot saved to: {plot_path}")

# Part 4, Q18: select all settings before evaluating the fixed test set.
# One final fit and one test predict per selected model; no test-based tuning.
final_performance = {}
final_predictions = {}
comparison_rows = []
for model_name, selected in selected_models.items():
    selected.fit(x_train, y_train)
    predictions = selected.predict(x_test)
    final_predictions[model_name] = predictions.tolist()
    final_performance[model_name] = (
        accuracy_score(y_test, predictions),
        f1_score(y_test, predictions, average='macro')
    )
    baseline_name = 'SVM' if model_name == 'SVM (RBF)' else model_name
    baseline = raw_performance[baseline_name]
    # Trees are left unscaled in selection; their control is the raw baseline.
    preprocessing = (baseline if model_name == 'Decision Tree'
                     else standardized_performance[baseline_name])
    final = final_performance[model_name]
    row = {
        'model': model_name,
        'selected_parameters': json.dumps(selected_parameters[model_name]),
        'baseline_accuracy': float(baseline[0]), 'baseline_macro_f1': float(baseline[1]),
        'preprocessing_only_accuracy': float(preprocessing[0]),
        'preprocessing_only_macro_f1': float(preprocessing[1]),
        'selected_accuracy': float(final[0]), 'selected_macro_f1': float(final[1]),
        'preprocessing_delta_macro_f1': float(preprocessing[1] - baseline[1]),
        'selection_delta_macro_f1': float(final[1] - preprocessing[1]),
        'total_delta_macro_f1': float(final[1] - baseline[1]),
    }
    comparison_rows.append(row)
    print(f"\n{model_name} | selected={selected_parameters[model_name]}")
    print(f"  Baseline: Accuracy={baseline[0]:.4f}, Macro-F1={baseline[1]:.4f}")
    print(f"  Preprocessing only: Accuracy={preprocessing[0]:.4f}, Macro-F1={preprocessing[1]:.4f}")
    print(f"  Selected test: Accuracy={final[0]:.4f}, Macro-F1={final[1]:.4f}")
    print(f"  Macro-F1 changes: preprocessing={row['preprocessing_delta_macro_f1']:+.4f}, "
          f"selection={row['selection_delta_macro_f1']:+.4f}")

output_dir = Path(__file__).resolve().parent
for filename, rows in [('part4_cv_results.csv', cv_rows),
                       ('part4_test_comparison.csv', comparison_rows)]:
    with (output_dir / filename).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
part4_result = {
    'protocol': '80/20 stratified split, seed=42; 5-fold StratifiedKFold on train only',
    'selection_rule': 'max mean Macro-F1, then min std, then candidate order for exact ties',
    'train_samples': len(y_train), 'test_samples': len(y_test),
    'cv_results': cv_results, 'selected_parameters': selected_parameters,
    'comparison': comparison_rows,
    'test_targets': y_test.tolist(), 'test_predictions': final_predictions,
    'reference_models_unchanged': {
        name: {'accuracy': float(raw_performance[name][0]),
               'macro_f1': float(raw_performance[name][1])}
        for name in ('Random Forest', 'Gaussian NB')
    },
    'interpretation_note': 'Sequential differences on this fixed split, not a universal causal decomposition.'
}
(output_dir / 'part4_results.json').write_text(
    json.dumps(part4_result, ensure_ascii=False, indent=2), encoding='utf-8'
)

#Part 5
data2 = load_digits()
num_samples, num_features = data2.data.shape
print('-' * 30)
print(f"Number of samples: {num_samples}")
print(f"Number of input features: {num_features}")
print("Class counts:")
for class_id, class_name in enumerate(data2.target_names):
    count = (data2.target == class_id).sum()
    print(f"  {class_name}: {count}")
print('-' * 30)

x_train, x_test, y_train, y_test = train_test_split(data2.data, data2.target, 
        test_size=0.2, random_state=42, stratify=data2.target)
std_x_train = scaler.fit_transform(x_train)
std_x_test = scaler.transform(x_test)

acc = {}
max_accuracy = 0
best_model = None

#decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Decision Tree Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
acc["Decision Tree"] = accuracy_score(y_test, y_pred)

#random forest classifier
random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Random Forest Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
acc["Random Forest"] = accuracy_score(y_test, y_pred)

#kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(std_x_train, y_train)
y_pred = knn.predict(std_x_test)
print(f"kNN Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"kNN Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
acc["kNN"] = accuracy_score(y_test, y_pred)

#Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print(f"Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Gaussian Naive Bayes Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
acc["Gaussian Naive Bayes"] = accuracy_score(y_test, y_pred)

#SVM classifier
svm = SVC(kernel='rbf', random_state=42)
svm.fit(std_x_train, y_train)
y_pred = svm.predict(std_x_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"SVM Macro-f1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
acc["SVM"] = accuracy_score(y_test, y_pred)

# Plot the confusion matrix for the classifier with the highest accuracy.
best_model = max(acc, key=acc.get)
max_accuracy = acc[best_model]
models = {
    "Decision Tree": (decision_tree, x_test),
    "Random Forest": (random_forest, x_test),
    "kNN": (knn, std_x_test),
    "Gaussian Naive Bayes": (gnb, x_test),
    "SVM": (svm, std_x_test),
}
best_classifier, best_x_test = models[best_model]
best_y_pred = best_classifier.predict(best_x_test)
cm = confusion_matrix(y_test, best_y_pred, labels=best_classifier.classes_)

fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(cm, cmap='Blues')
positions = range(len(best_classifier.classes_))
ax.set_xticks(positions, best_classifier.classes_)
ax.set_yticks(positions, best_classifier.classes_)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(col, row, str(cm[row, col]), ha='center', va='center',
                color='white' if cm[row, col] > cm.max() / 2 else 'black')
ax.set_title(f'{best_model} | Accuracy: {max_accuracy:.4f}')
fig.tight_layout()
confusion_matrix_path = Path(__file__).resolve().parent / 'best_classifier_confusion_matrix.png'
fig.savefig(confusion_matrix_path, dpi=200, bbox_inches='tight')
print(f'\nBest classifier: {best_model}, Accuracy: {max_accuracy:.4f}')
print(f'Confusion matrix saved to: {confusion_matrix_path}')
plt.show()
plt.close(fig)

std_x_train = scaler.fit_transform(x_train)
std_x_test = scaler.transform(x_test)
knn.fit(std_x_train, y_train)
std_y_pred = knn.predict(std_x_test)
print(f"kNN (standardized) Accuracy: {accuracy_score(y_test, std_y_pred):.4f}")
print(f"kNN (standardized) Macro-f1 Score: {f1_score(y_test, std_y_pred, average='macro'):.4f}")