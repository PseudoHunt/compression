from sklearn.linear_model import Ridge

def add_batch_update_u(self, inp, out):
    inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2]).cpu().numpy()
    outs = out.view(out.shape[0] * out.shape[1], out.shape[2]).cpu().numpy()

    # Projection into compressed space
    V = self.truc_v.cpu().numpy()
    Sigma = self.truc_sigma.cpu().numpy()
    x = inps @ V.T @ Sigma  # [N, rank]

    # Sample weights from Wanda++ mask (average across output dim if needed)
    sample_weights = self.wanda_mask.mean(dim=0).cpu().numpy() if self.wanda_mask is not None else None

    # Ridge regression
    model = Ridge(alpha=1e-3, fit_intercept=False)
    model.fit(x, outs, sample_weight=sample_weights)
    updated_uT_np = model.coef_.T  # shape: (out_dim, rank)

    # Convert back to tensor
    self.updated_uT = torch.tensor(updated_uT_np).to(self.truc_sigma.device, dtype=torch.float32)

    # Optional: compute updated error for tracking
    updated_output = x @ self.updated_uT.cpu().numpy()
    error = np.linalg.norm(outs - updated_output) / np.linalg.norm(outs)
    self.updated_error = float(error)
