from classification_project.model import TimmModel
from classification_project.dataset import MNIST
from classification_project.predict import check_accuracy
from classification_project.engine import train_model
import hydra
from omegaconf import DictConfig


@hydra.main(config_path=r"..\configs", config_name="config")
def run(cfg: DictConfig):
    model = TimmModel(model_name=cfg.model.model.model_name[2], pretrained=cfg.model.model.pretrained,
                      learning_rate=cfg.model.model.learning_rate, batch_size=cfg.model.model.batch_size,
                      num_epochs=cfg.model.model.num_epochs)
    mnist = MNIST(root=cfg.datasets.ds.root, batch_size=cfg.datasets.ds.batch_size)

    train_loader = mnist.train_loader()
    test_loader = mnist.test_loader()

    train_model(train_loader, model, cfg.logging.wandb)

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)


if __name__ == '__main__':
    run()
