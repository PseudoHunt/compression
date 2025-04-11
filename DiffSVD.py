import torch
import torch.nn as nn
import torch.nn.functional as F

class RankConstrainedSVD(nn.Module):
    def __init__(self, weight_matrix, target_ratio=0.3, temperature=10.0, alpha=1.0):
        """
        Args:
            weight_matrix: torch.Tensor of shape (d, d)
            target_ratio: desired ratio of retained singular values (e.g., 0.3)
            temperature: temperature for sigmoid activation
            alpha: weight for the rank penalty term
        """
        super().__init__()
        self.register_buffer("W", weight_matrix)
        
        # Decompose once and keep U, S, Vh fixed
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
        self.register_buffer("U", U)
        self.register_buffer("Vh", Vh)
        self.register_buffer("S_orig", S)

        # Learnable binary mask (soft)
        self.M = nn.Parameter(torch.randn_like(S))

        self.temperature = temperature
        self.alpha = alpha
        self.d = weight_matrix.shape[0]
        self.target_rank = int(target_ratio * self.d)

    def forward(self):
        # Soft binary mask
        mask = torch.sigmoid(self.M * self.temperature)

        # Masked singular values
        S_masked = self.S_orig * mask
        S_matrix = torch.diag(S_masked)

        # Reconstructed weight
        W_reconstructed = self.U @ S_matrix @ self.Vh

        # Soft rank penalty
        rank_penalty = ((mask.sum() - self.target_rank) ** 2)

        return W_reconstructed, mask, rank_penalty

    def hard_threshold(self):
        """
        Returns the weight matrix with a hard top-k truncation.
        """
        with torch.no_grad():
            mask = torch.sigmoid(self.M * self.temperature)
            topk = torch.topk(mask, self.target_rank)
            hard_mask = torch.zeros_like(mask)
            hard_mask[topk.indices] = 1.0
            S_masked = self.S_orig * hard_mask
            S_matrix = torch.diag(S_masked)
            W_hard = self.U @ S_matrix @ self.Vh
        return W_hard, hard_mask
