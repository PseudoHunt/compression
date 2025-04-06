def add_batch_update_u(self, inp, out, mask=None):
    inps = inp.view(-1, inp.shape[2])     # (N, d)
    outs = out.view(-1, out.shape[2])     # (N, D)
    Z = torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)  # (N, r)

    N, D = outs.shape
    r = self.truc_sigma.shape[0]
    U = torch.zeros((r, D), device=inps.device)

    if mask is None:
        # Full least squares fallback
        self.updated_uT = torch.linalg.lstsq(Z, outs).solution
        return

    mask = mask.view(-1, D)
    rows, cols = (mask > 0).nonzero(as_tuple=True)  # indexes of important elements
    y_vals = outs[rows, cols]
    z_vals = Z[rows]  # (num_masked, r)

    # For each output dimension (column of U), solve only if that column has any mask
    for j in range(D):
        sel = (cols == j)
        if sel.sum() == 0:
            continue
        Z_j = z_vals[sel]     # (m_j, r)
        y_j = y_vals[sel]     # (m_j,)
        A = Z_j.T @ Z_j + 1e-6 * torch.eye(r, device=inps.device)
        b = Z_j.T @ y_j
        U[:, j] = torch.linalg.solve(A, b)

    self.updated_uT = U
