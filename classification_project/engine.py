import wandb


def train_model(train_loader, model, cfg_dict):
    wandb.init(project=cfg_dict['project_name'])
    wandb.config = {'learning_rate': model.lr, 'epochs': model.num_epochs, 'batch_size': model.batch_size}
    wandb.watch(model, criterion=model.criterion, log=cfg_dict['log'], log_freq=cfg_dict['log_freq'])

    example_ct = 0  # number of examples seen
    batch_ct = 0

    for epoch in range(1, model.num_epochs + 1, 1):
        _train_log(0, 0, epoch)

        print(f'epoch:{epoch}')

        for batch, (data, targets) in enumerate(train_loader):

            data = data.to(device=model.device)
            targets = targets.to(device=model.device)

            example_ct += len(data)
            batch_ct += 1

            if not model.model_name.startswith('vit'):
                data = data.repeat(1, 3, 1, 1)
            else:
                data = data.repeat(1, 3, 8, 8)

            scores = model.model(data)
            loss = model.criterion(scores, targets)

            if ((batch + 1) % 25) == 0:
                _train_log(loss, example_ct, epoch)

            model.optimizer.zero_grad()
            loss.backward()

            model.optimizer.step()


def _train_log(loss, example_ct, epoch):
    wandb.log({'epoch': epoch, 'loss': loss}, step=example_ct)
    print(f'Loss after ' + str(example_ct).zfill(5) + f' examples: {loss:.3f}')
