def train_model(train_loader, model):
    for epoch in range(1, model.num_epochs + 1, 1):

        print(f'epoch:{epoch}')

        for batch, (data, targets) in enumerate(train_loader):

            data = data.to(device=model.device)
            targets = targets.to(device=model.device)

            if not model.model_name.startswith('vit'):
                data = data.repeat(1, 3, 1, 1)
            else:
                data = data.repeat(1, 3, 8, 8)

            scores = model.model(data)
            loss = model.criterion(scores, targets)

            model.optimizer.zero_grad()
            loss.backward()

            model.optimizer.step()
