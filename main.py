import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from attacks import fgsm_attack, pgd_attack
from defense import adversarial_training, gradient_masking_training
from model import Net, RNNNet
from utils import test_model, evaluate_robustness, visualize_adversarial_examples

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms & loaders
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset, batch_size=1000, shuffle=False)

# Instantiate baseline models
cnn = Net().to(device)
rnn = RNNNet().to(device)

# Loss and optimizers
criterion = nn.CrossEntropyLoss()
opt_cnn = torch.optim.Adam(cnn.parameters(), lr=0.001)
opt_rnn = torch.optim.Adam(rnn.parameters(), lr=0.001)

# Baseline training
for epoch in range(5):
    cnn.train(); rnn.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        # Train CNN
        opt_cnn.zero_grad()
        loss_c = criterion(cnn(images), labels)
        loss_c.backward()
        opt_cnn.step()
        # Train RNN
        opt_rnn.zero_grad()
        loss_r = criterion(rnn(images), labels)
        loss_r.backward()
        opt_rnn.step()

# Evaluate baseline
print("\n[Baseline CNN]")
test_model(cnn, testloader, device)
print("[Baseline RNN]")
test_model(rnn, testloader, device)

# Pre-defense robustness
print("\n[Pre-defense Robustness CNN]")
evaluate_robustness(cnn, testloader, device, attack_fn=fgsm_attack, epsilon=0.25)
evaluate_robustness(cnn, testloader, device, attack_fn=pgd_attack, epsilon=0.25)
print("[Pre-defense Robustness RNN]")
evaluate_robustness(rnn, testloader, device, attack_fn=fgsm_attack, epsilon=0.25)
evaluate_robustness(rnn, testloader, device, attack_fn=pgd_attack, epsilon=0.25)

# Defense: Adversarial Training
print("\n[Adversarial Training CNN]")
cnn_adv = adversarial_training(Net().to(device), trainloader, device, pgd_attack, epsilon=0.25)

print("[Adversarial Training RNN]")
rnn_adv = adversarial_training(RNNNet().to(device), trainloader, device, pgd_attack, epsilon=0.25)

# Defense: Gradient Masking (optional)
print("\n[Gradient Masking Training CNN]")
cnn_masked = gradient_masking_training(Net().to(device), trainloader, device, quantization_levels=16)

print("\n[Gradient Masking Training RNN]")
rnn_masked = gradient_masking_training(RNNNet().to(device), trainloader, device, quantization_levels=16)

# Post-defense evaluation
print("\n[Post-AdvTrain Robustness CNN]")
test_model(cnn_adv, testloader, device)
evaluate_robustness(cnn_adv, testloader, device, attack_fn=fgsm_attack, epsilon=0.25)
evaluate_robustness(cnn_adv, testloader, device, attack_fn=pgd_attack, epsilon=0.25)

print("[Post-AdvTrain Robustness RNN]")
test_model(rnn_adv, testloader, device)
evaluate_robustness(rnn_adv, testloader, device, attack_fn=fgsm_attack, epsilon=0.25)
evaluate_robustness(rnn_adv, testloader, device, attack_fn=pgd_attack, epsilon=0.25)

print("\n[Post-GradMask Robustness CNN]")
test_model(cnn_masked, testloader, device)
evaluate_robustness(cnn_masked, testloader, device, attack_fn=fgsm_attack, epsilon=0.25)
evaluate_robustness(cnn_masked, testloader, device, attack_fn=pgd_attack, epsilon=0.25)

print("[Post-GradMask Robustness RNN]")
test_model(rnn_masked, testloader, device)
evaluate_robustness(rnn_masked, testloader, device, attack_fn=fgsm_attack, epsilon=0.25)
evaluate_robustness(rnn_masked, testloader, device, attack_fn=pgd_attack, epsilon=0.25)

# Visualize adversarial examples for CNN
visualize_adversarial_examples(cnn, testloader, device, epsilon=0.45, name= "cnn_adv")
# Visualize adversarial examples for RNN
visualize_adversarial_examples(rnn, testloader, device, epsilon=0.45, name= "rnn_adv")
