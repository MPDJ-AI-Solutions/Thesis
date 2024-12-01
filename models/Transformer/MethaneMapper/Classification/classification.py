import torch
import torch.nn as nn

class ClassifierPredictor(nn.Module):
    def __init__(
            self, num_classes: int,
            embedding_dim: int,
            threshold: float = 0.5,
            num_heads: int = 8
    ):
        """
        A classification-specific version of the predictor module.

        Args:
            num_classes (int): Number of classes for the classification task.
            embedding_dim (int): Dimension of the feature embeddings.
            threshold (float): Threshold for class predictions.
            num_heads (int): Number of attention heads if needed in the architecture.
        """
        super(ClassifierPredictor, self).__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.threshold = threshold

        # FFN for classification logits
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, num_classes)  # Output: class logits
        )

        # Optional attention mechanism for refining predictions (if needed)
        self.pre_attention_norm = nn.LayerNorm(embedding_dim)
        self.attention_layer = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, batch_first=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, e_out: torch.Tensor, fe: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            e_out (torch.Tensor): Output from the encoder, shape (batch_size, seq_length, embedding_dim).
            fe (torch.Tensor, optional): Additional feature embeddings for attention, shape (batch_size, seq_length, embedding_dim).

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes).
        """
        # Optional: Apply attention refinement
        if fe is not None:
            normed_e_out = self.pre_attention_norm(e_out)
            normed_fe = self.pre_attention_norm(fe)
            attention_output, _ = self.attention_layer(normed_e_out, normed_fe, normed_fe)
            e_out = e_out + attention_output  # Residual connection

        # Apply classification head
        class_logits = self.classification_head(e_out.mean(dim=1))  # Pool sequence to a single vector per batch

        return class_logits