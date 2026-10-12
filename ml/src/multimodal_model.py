import torch
import torch.nn as nn


class SatelliteEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class EnvironmentalEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class MultimodalFloodClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.satellite_encoder = SatelliteEncoder()
        self.environmental_encoder = EnvironmentalEncoder()

        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, 2),
        )

    def forward(self, satellite, environmental):

        satellite_features = self.satellite_encoder(
            satellite
        )

        environmental_features = self.environmental_encoder(
            environmental
        )

        combined = torch.cat(
            [
                satellite_features,
                environmental_features,
            ],
            dim=1,
        )

        output = self.classifier(combined)

        return output


if __name__ == "__main__":

    print("NEXUS Multimodal Model")
    print("======================")

    model = MultimodalFloodClassifier()

    satellite = torch.randn(2, 2, 512, 512)
    environmental = torch.randn(2, 2)

    with torch.no_grad():
        output = model(
            satellite,
            environmental,
        )

    print("Satellite input:     ", satellite.shape)
    print("Environmental input: ", environmental.shape)
    print("Model output:        ", output.shape)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Parameters:           ", parameters)