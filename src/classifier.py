import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size=512, num_classes=2):
        super().__init__()

        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.classifier(x)


# Create classifier
model = Classifier()

# Simulated Temporal Attention output: (batch_size, 512)
context = torch.randn(1, 512)

# Get class scores
output = model(context)

print("Context shape:", context.shape)
print("Classifier output shape:", output.shape)
print("Class scores:", output)

# Predict the class
predicted_class = torch.argmax(output, dim=1).item()
label = "Fake" if predicted_class == 1 else "Real"

print("Predicted class:", label)
