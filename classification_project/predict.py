import torch


def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy o test data')
    num_correct = 0
    num_samples = 0
    model.model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=model.device)
            y = y.to(device=model.device)

            if not model.model_name.startswith('vit'):
                x = x.repeat(1, 3, 1, 1)
            else:
                x = x.repeat(1, 3, 8, 8)

            scores = model.model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f'Got {num_correct}/{num_samples} with accuracy {(float(num_correct) / float(num_samples)) * 100}')

    model.model.train()
