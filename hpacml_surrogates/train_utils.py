import torch


class EarlyStopper:
    def __init__(self, patience=1, min_delta_percent=1):
        self.patience = patience
        self.min_delta_percent = min_delta_percent / 100.0
        self.counter = 0
        self.best_loss = None 

    def early_stop(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
            return False

        improvement = (self.best_loss - validation_loss) / self.best_loss
        if improvement >= self.min_delta_percent:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def MAPE(actual, forecast):
    if actual.shape != forecast.shape:
        raise ValueError("The shape of actual and forecast tensors must be the same.")

    epsilon = 1e-8  # small constant to avoid division by zero
    mape = torch.mean(torch.abs((actual - forecast) / (actual + epsilon))) * 100
    return mape


def train_loop(dataloader, model, loss_fn, optimizer, epoch, device):
    size = len(dataloader.dataset)
    model = model.to(device)
    test_loss = 0
    for batch, dat in enumerate(dataloader):
        X = dat[0]
        y = dat[1]
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        test_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(X)
            print(f"(Training) Epoch: {epoch}, loss: {test_loss/len(dataloader):>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device, user_loss_fn=None):
    test_loss = 0
    test_loss_user = 0
    num_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            if user_loss_fn is not None:
                test_loss_user += user_loss_fn(pred, y).item()
            num_batches += 1

    print(f"Test Error: \n Avg loss: {test_loss / num_batches:>8f}")
    print(f"Test Error: \n Avg loss (user): {test_loss_user / num_batches:>8f}")
    return test_loss / num_batches, test_loss_user / num_batches
