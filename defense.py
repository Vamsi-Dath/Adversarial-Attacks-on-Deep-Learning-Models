import torch
import torch.nn as nn
import torch.optim as optim
from attacks import fgsm_attack, pgd_attack


def adversarial_training(model, trainloader, device, attack_fn, epsilon, epochs=3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(f"[AdvTrain] Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            # Generate adversarial examples
            if attack_fn.__name__ == "pgd_attack":
                adv_images = pgd_attack(model, images, labels, epsilon=epsilon)
            else:
                images.requires_grad = True
                loss_clean = criterion(model(images), labels)
                grad = torch.autograd.grad(loss_clean, images)[0]
                adv_images = fgsm_attack(images, epsilon, grad)
            # Training step
            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:
                print(f"  batch {batch_idx}/{len(trainloader)}")
    return model


def gradient_masking_training(model, trainloader, device, quantization_levels=16, epochs=3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print(f"[GradMask] Epoch {epoch+1}/{epochs}")
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            # Quantize inputs to mask small perturbations
            images_q = torch.round(images * (quantization_levels - 1)) / (quantization_levels - 1)
            optimizer.zero_grad()
            outputs = model(images_q)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:
                print(f"  batch {batch_idx}/{len(trainloader)}")
    return model