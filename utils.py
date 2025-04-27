import torch
import torch.nn as nn
from attacks import fgsm_attack,pgd_attack

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

def evaluate_robustness(model, dataloader, device, attack_fn, epsilon):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True
        adv_images = attack_fn(model, images, labels, epsilon) \
            if attack_fn.__name__ == "pgd_attack" else \
            fgsm_attack(images, epsilon, torch.autograd.grad(
                nn.CrossEntropyLoss()(model(images), labels),
                images,
                retain_graph=True
            )[0])
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Robust Accuracy under {attack_fn.__name__}: {100 * correct / total:.2f}%")
import matplotlib.pyplot as plt

def visualize_adversarial_examples(model, dataloader, device, epsilon, num_samples=3, name=""):
    model.eval()
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    fig, axs = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    for idx in range(num_samples):
        image = images[idx:idx+1].to(device)
        label = labels[idx:idx+1].to(device)

        # Clean prediction
        output = model(image)
        _, pred_clean = torch.max(output, 1)

        # FGSM Attack
        image.requires_grad = True
        output = model(image)
        loss = nn.CrossEntropyLoss()(output, label)
        model.zero_grad()
        loss.backward()
        data_grad = image.grad.data
        fgsm_img = fgsm_attack(image, epsilon, data_grad)
        fgsm_output = model(fgsm_img)
        _, pred_fgsm = torch.max(fgsm_output, 1)

        # PGD Attack
        from attacks import pgd_attack
        pgd_img = pgd_attack(model, image, label, epsilon=epsilon, alpha=0.01, iters=50)
        pgd_output = model(pgd_img)
        _, pred_pgd = torch.max(pgd_output, 1)

        titles = [f"Clean (Pred: {pred_clean.item()})", f"FGSM (Pred: {pred_fgsm.item()})", f"PGD (Pred: {pred_pgd.item()})"]
        images_to_plot = [image, fgsm_img, pgd_img]

        for j in range(3):
            ax = axs[idx, j] if num_samples > 1 else axs[j]
            img_np = images_to_plot[j].detach().cpu().squeeze().numpy()
            ax.imshow(img_np, cmap='gray')
            ax.set_title(titles[j])
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    print(f"[INFO] Saved visualization with {num_samples} examples as '{name}.png'")
