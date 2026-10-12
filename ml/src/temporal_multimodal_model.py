import torch
import torch.nn as nn


class SatelliteEncoder(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
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

            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)


class TemporalEncoder(nn.Module):
    def __init__(self, input_size=28):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),

            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )

    def forward(self, x):
        return self.network(x)


class TemporalMultimodalFloodClassifier(nn.Module):
    def __init__(
        self,
        satellite_channels=2,
        temporal_features=28,
        num_classes=2,
    ):
        super().__init__()

        self.satellite_encoder = SatelliteEncoder(
            in_channels=satellite_channels
        )

        self.temporal_encoder = TemporalEncoder(
            input_size=temporal_features
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),

            nn.Linear(64, num_classes),
        )

    def forward(self, satellite, temporal):
        satellite_features = self.satellite_encoder(satellite)
        temporal_features = self.temporal_encoder(temporal)

        fused = torch.cat(
            [satellite_features, temporal_features],
            dim=1,
        )

        return self.classifier(fused)


if __name__ == "__main__":
    model = TemporalMultimodalFloodClassifier()

    satellite = torch.randn(2, 2, 512, 512)
    temporal = torch.randn(2, 28)

    with torch.no_grad():
        output = model(satellite, temporal)

    print("Temporal Multimodal Model Test")
    print("==============================")
    print("Satellite input:", satellite.shape)
    print("Temporal input:", temporal.shape)
    print("Output:", output.shape)

    parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Parameters:", parameters)