import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix


def evaluate_performance(predictions, true_labels):
    # Convert predictions and true_labels to integers.
    predictions = predictions.astype(int)
    true_labels = true_labels.astype(int)

    accuracy = accuracy_score(true_labels, predictions)

    cm = confusion_matrix(true_labels, predictions)
    # Output the confusion matrix

    print("Confusion matrix:")
    print(cm)

    # Initialization result dictionary
    results = {
        'accuracy': accuracy,
        'class_metrics': {},
    }

    report = classification_report(true_labels, predictions, output_dict=True)

    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1_score = report['macro avg']['f1-score']

    class_labels = np.unique(np.concatenate((true_labels, predictions)))
    for label in class_labels:
        # get  TP, FP, TN, FN
        TP = cm[label, label]
        FP = cm[:, label].sum() - TP
        FN = cm[label, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)


        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # get fpr
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

        results['class_metrics'][label] = \
            {
                'precision': round(precision, 8),
                'recall': round(recall, 8),
                'f1_score': round(f1_score, 8),
                'accuracy': round(TP / (TP + FP), 8),
                'fpr': round(fpr, 8),
                'support': report[str(label)]['support'],
            }

    # print("Classification Report:")
    # print(formatted_report)
    print("Class Metrics:")
    for label, metrics in results['class_metrics'].items():
        print(f"Class {label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    return accuracy, macro_precision, macro_recall, macro_f1_score, results


def print_meetrics(a, accuracy, macro_precision, macro_recall, macro_f1_score, results, Accuracy=None, Precision=None,
                   Recall=None, F1_score=None,
                   C1_Recall=None, C1_F1_score=None, FPR=None):
    print(f'Accuracy: {accuracy:.8f}')
    # print(f'False Alarm Rate: {false_alarm_rate:.5f}')
    print(f'Macro Precision: {macro_precision:.8f}')
    print(f'Macro Recall: {macro_recall:.8f}')
    print(f'Macro F1 Score: {macro_f1_score:.8f}')

    # # 如果没有提供数组，则创建新的数组
    # if Accuracy is None:
    #     Accuracy = np.zeros(120)
    #     Precision = np.zeros(120)
    #     Recall = np.zeros(120)
    #     F1_score = np.zeros(120)
    #     C1_Recall = np.zeros(120)
    #     C1_F1_score = np.zeros(120)
    #     FPR = np.zeros(120)

    Accuracy[a] = accuracy
    Precision[a] = macro_precision
    Recall[a] = macro_recall
    F1_score[a] = macro_f1_score
    target_metrics = {
        'recall': None,
        'f1_score': None,
        'fpr': None
    }

    if 'class_metrics' in results and 1 in results['class_metrics']:
        class_1 = results['class_metrics'][1]
        target_metrics.update({
            'recall': class_1.get('recall'),
            'f1_score': class_1.get('f1_score'),
            'fpr': class_1.get('fpr')
        })

    C1_Recall[a] = target_metrics['recall']
    C1_F1_score[a] = target_metrics['f1_score']
    FPR[a] = target_metrics['fpr']

    print('The prediction accuracy is：', Accuracy[a] * 100, '%')
    print("----------")
    return Accuracy, Precision, Recall, F1_score, C1_Recall, C1_F1_score, FPR
