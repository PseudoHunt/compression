# ✅ INSERT THIS FUNCTION ANYWHERE AT TOP-LEVEL IN THE SCRIPT

def wanda_importance_scores(W, X):
    """
    Compute Wanda-style importance score for each output neuron (row of W)
    W: [D_out, D_in] - weight matrix of a linear layer
    X: [N, D_in] - input activations collected during forward pass
    Returns:
        importance_scores: [D_out] - per-output importance
    """
    x_norms = X.norm(p=2, dim=0)  # [D_in]
    importance = (W.abs() * x_norms).sum(dim=1)  # [D_out]
    return importance


def update_U_with_weighted_least_squares(Y, Z, importance_scores):
    """
    Weighted least squares update of U^T in SVD-LLM
    Y: [N, D_out] - target output (WX)
    Z: [N, rank] - compressed input (Sigma @ V^T @ X)
    importance_scores: [D_out] - Wanda-style importance
    Returns:
        updated_uT: [rank, D_out]
    """
    weights = torch.sqrt(importance_scores / (importance_scores.max() + 1e-6)).clamp(min=1e-4).to(Y.device)
    W_diag = weights.view(1, -1)  # [1, D_out]
    Y_weighted = Y * W_diag
    updated_uT = torch.linalg.lstsq(Z, Y_weighted).solution.T  # [rank, D_out]
    return updated_uT
