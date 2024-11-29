from torch import nn
from transformers import DetrConfig, DetrForSegmentation, DetrForObjectDetection


class CustomDetr(nn.Module):
    def __init__(self, detr_model_name="facebook/detr-resnet-50", num_channels=9):
        super().__init__()

        # Load pre-trained DETR model
        config = DetrConfig.from_pretrained(detr_model_name)
        config.num_labels = 2  # One foreground class + background
        config.num_queries = 10
        config.use_masks = True

        self.detr = DetrForSegmentation(config=config)

        # Modify the first convolutional layer of the backbone to accept 9 channels
        # Access the backbone
        backbone = self.detr.detr.model.backbone

        # Modify the first convolutional layer
        conv1 = backbone.conv_encoder.model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias,
        )

        # Replace the original conv1 with the new one
        backbone.conv_encoder.model.conv1 = new_conv1

        # Freeze backbone layers except the first conv layer
        for name, param in backbone.named_parameters():
            if "conv1" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, pixel_values, labels):
        return self.detr(pixel_values, labels = labels)


class CustomDetrForClassification(nn.Module):
    def __init__(self, detr_model_name="facebook/detr-resnet-50", num_channels=9, num_classes=2):
        super().__init__()

        # Load pre-trained DETR model
        config = DetrConfig.from_pretrained(detr_model_name)
        config.num_labels = num_classes  # Number of classification labels
        config.use_decoder = True  # Ensure the decoder is retained for processing queries
        config.output_hidden_states = True  # Ensure hidden states are returned
        self.detr = DetrForObjectDetection(config=config)

        # Modify the first convolutional layer of the backbone to accept 9 channels
        backbone = self.detr.model.backbone
        conv1 = backbone.conv_encoder.model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias,
        )

        # Replace the original conv1 with the new one
        backbone.conv_encoder.model.conv1 = new_conv1

        # Freeze backbone layers except the first conv layer
        for name, param in backbone.named_parameters():
            if "conv1" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Add a classification head to process the outputs of the decoder
        hidden_size = config.d_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        # Pass inputs through DETR backbone and transformer
        outputs = self.detr.model(pixel_values)

        # Extract decoder output (shape: batch_size, num_queries, d_model)
        decoder_output = outputs.decoder_hidden_states[-1]

        # Apply classification head (average over all queries)
        logits = self.classifier(decoder_output.mean(dim=1))  # (batch_size, num_classes)

        return logits
