from classification_project.model import TimmModel
from classification_project.dataset import MNIST
from classification_project.predict import check_accuracy
from classification_project.engine import train_model


def run():
    model = TimmModel('vit_tiny_patch16_224')
    mnist = MNIST()

    train_loader = mnist.train_loader()
    test_loader = mnist.test_loader()

    train_model(train_loader, model)

    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)


if __name__ == '__main__':
    run()
