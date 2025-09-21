import torch
from torch.utils.data import DataLoader
from srcs.Train.data_management import CSVDatasetF3
from srcs.Train.model_config import ImageMLP
from srcs.Train.train_loop import train

def main():
    # get training data from dataset.csv 
    training_data = CSVDatasetF3("train", "test.csv", root=".")
    evaluation_data = CSVDatasetF3("val", "test.csv", root=".")
    print("training data get it")

    train_loader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    evaluation_loader = DataLoader(evaluation_data, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    print("data loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageMLP(in_shape=(3,256,256), num_classes=len(training_data.class_to_idx)).to(device)
    print("model setup")

    # FIXED CALL
    train(
        model=model,
        train_loader=train_loader,
        eval_loader=evaluation_loader,
        device=device,
        epochs=10,
        lr=1e-3
    )
    print("training done...")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()