import torch
import torch.nn as nn

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(model, images, labels, epsilon=0.25, alpha=0.01, iters=40):
    images = images.clone().detach().to(images.device)
    original_images = images.clone().detach()
    images.requires_grad = True

    for _ in range(iters):
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data
        images = images + alpha * grad.sign()
        eta = torch.clamp(images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()
        images.requires_grad = True
    return images
