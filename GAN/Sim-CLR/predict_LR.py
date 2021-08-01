import numpy as np
import torch
import torchvision

from logistic_model import Logistic_Model


def train(loader, model: Logistic_Model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)
        # print("output")
        # print(y)
        output = model(x)

        # print(output)

        loss = criterion(output, y)
        # print(model.linear.weight)
        # print(loss.requires_grad)
        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()
        # print("### Grad after Backprop ####")
        # print(next(model.backbone.parameters())[0][0][0])
        # print(next(model.linear.parameters())[0][0:10])

        loss_epoch += loss.item()
        if step % 10 == 0:
            print(
                f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
            )



    return loss_epoch, accuracy_epoch


def test(loader, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


if __name__ == "__main__":
    IMAGE_SIZE = 224
    DATA_ROOT = "../data/STL10"
    EPOCHS = 100
    LEARNING_RATE = 0.0003
    PROJECTION_DIM = 128
    BATCH_SIZE = 64
    TEMPERATURE = 0.5
    WEIGHT_DECAY = 1e-4
    SIM_CLR_MODEL_PATH = "sim-clr.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(IMAGE_SIZE),
        torchvision.transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.STL10(
        root=DATA_ROOT,
        split="train",
        download=False,
        transform=transforms
    )

    test_dataset = torchvision.datasets.STL10(
        root=DATA_ROOT,
        split="test",
        download=False,
        transform=transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
    )

    # sim_clr_model = SIM_CLR(projection_dim=PROJECTION_DIM)
    # sim_clr_model.load_state_dict(torch.load("sim-clr.pth", map_location=device))
    # sim_clr_model = sim_clr_model.to(device)
    # sim_clr_model.eval()
    n_classes = 10
    # n_features = sim_clr_model.projection_heads[0].out_features
    model = Logistic_Model(
        device=device,
        projection_dim=PROJECTION_DIM,
        n_classes=n_classes,
        path=SIM_CLR_MODEL_PATH
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # print("### Creating features from pre-trained context model ###")
    # (train_X, train_y, test_X, test_y) = get_features(
    #     sim_clr_model, train_loader, test_loader, device
    # )
    #
    # arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
    #     train_X, train_y, test_X, test_y, BATCH_SIZE
    # )

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch}=====>>")
        print("### Grad before Backprop ####")
        print(next(model.backbone.parameters())[0][0][0])
        print(next(model.linear.parameters())[0][0:10])
        loss_epoch, accuracy_epoch = train(
            train_loader, model, criterion, optimizer
        )
        print("### Grad after Backprop ####")
        print(next(model.backbone.parameters())[0][0][0])
        print(next(model.linear.parameters())[0][0:10])
        print("Summary")
        print(
            f"Epoch [{epoch}/{EPOCHS}]\t "
            f"Loss: {loss_epoch / len(train_loader)}\t "
            f"Accuracy: {accuracy_epoch / len(train_loader)}"
        )

    # final testing
    loss_epoch, accuracy_epoch = test(
        test_loader, model, criterion, optimizer
    )
    print(
        f"[FINAL]\t "
        f"Loss: {loss_epoch / len(test_loader)}\t "
        f"Accuracy: {accuracy_epoch / len(test_loader)}"
    )
