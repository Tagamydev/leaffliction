import torch
import torch.nn.functional as F
from tabulate import tabulate

def print_test_result(t_loss, e_loss, t_accuracy, e_accuracy, t_sens, e_sens, t_spec, e_spec):
    """
    Prints training vs evaluation metrics in a nice table.
    """
    headers = ["Metric", "Train", "Eval"]
    table = [
        ["Loss", f"{t_loss:.4f}", f"{e_loss:.4f}"],
        ["Accuracy", f"{t_accuracy:.4f}", f"{e_accuracy:.4f}"],
        ["Sensitivity", f"{t_sens:.4f}", f"{e_sens:.4f}"],
        ["Specificity", f"{t_spec:.4f}", f"{e_spec:.4f}"],
    ]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def loss(model, dataloader, device="cpu"):
    """
    Computes average cross-entropy loss for a dataset.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            batch_loss = criterion(logits, y)
            total_loss += batch_loss.item() * x.size(0)
            total_samples += x.size(0)

    return total_loss / total_samples if total_samples > 0 else float("nan")


def calculate_metrics(model, dataloader, device="cpu"):
    model.eval()

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["image"].to(device)
            y_true = batch["y"].to(device)

            # Forward pass
            logits = model(x)
            y_pred = torch.argmax(logits, dim=1)

            # Update confusion matrix counts
            for yt, yp in zip(y_true.cpu().tolist(), y_pred.cpu().tolist()):
                if yt == 1 and yp == 1:
                    true_positives += 1
                elif yt == 0 and yp == 0:
                    true_negatives += 1
                elif yt == 0 and yp == 1:
                    false_positives += 1
                elif yt == 1 and yp == 0:
                    false_negatives += 1
                total += 1

    # Metrics
    accuracy = (true_positives + true_negatives) / max(1, total)
    sensitivity = true_positives / max(1, (true_positives + false_positives))
    specificity = true_negatives / max(1, (true_negatives + false_negatives))

    return accuracy, sensitivity, specificity


def testing(model, train_loader, eval_loader, device="cpu"):
    """
    Run evaluation on training and eval sets, compute loss + metrics.
    """
    print("Running tests...")

    # Loss
    t_loss = loss(model, train_loader, device)
    e_loss = loss(model, eval_loader, device)

    # Metrics
    t_accuracy, t_sens, t_spec = calculate_metrics(model, train_loader, device)
    e_accuracy, e_sens, e_spec = calculate_metrics(model, eval_loader, device)

    # Results
    print_test_result(t_loss, e_loss, t_accuracy, e_accuracy, t_sens, e_sens, t_spec, e_spec)

