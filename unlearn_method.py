import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset

class UnlearnMethod:
    def __init__(self, model, loss_function, save_path, args) -> None:
        self.unlearn_dataloaders = None
        self.model = model
        self.loss_function = loss_function
        self.save_path = save_path
        self.args = args 
        self.params = {}

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders
        return

    def get_unlearned_model(self) -> nn.Module:
        return self.model 

    def get_params(self) -> dict:
        return self.params

class UnLearnDataset(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            data = self.forget_data[index]
            unlearn_label = 1
            return data, unlearn_label
        else:
            data = self.retain_data[index - self.forget_len]
            unlearn_label = 0
            return data, unlearn_label

class SimpleDataset(Dataset):
    def __init__(self, data, targets) -> None:
        super().__init__()
        self.data = data 
        self.targets = targets 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (self.data[index], self.targets[index])
    


def get_unlearn_method(name):

    # def retrain_method(loaders, model, criterion, args):
    def retrain_method(loaders, model, criterion, args, mask=None):

        retain_loader = loaders["retain"]

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.unlearn_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        model.train()

        for epoch in range(args.unlearn_epochs):

            print(f"Retrain Epoch {epoch}")

            for images, targets in retain_loader:

                images = images.cuda()
                targets = targets.cuda()

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(outputs, targets)

                loss.backward()

                optimizer.step()

    return retrain_method


def save_unlearn_checkpoint(model, evaluation_result, args):

    os.makedirs(args.save_dir, exist_ok=True)

    path = os.path.join(
        args.save_dir,
        "unlearn_checkpoint.pth"
    )

    torch.save(
        {
            "model": model.state_dict(),
            "evaluation_result": evaluation_result,
        },
        path,
    )


def load_unlearn_checkpoint(model, device, args):

    path = os.path.join(
        args.save_dir,
        "unlearn_checkpoint.pth"
    )

    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    return model, checkpoint["evaluation_result"]