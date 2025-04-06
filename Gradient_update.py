def add_batch_update_u(self, inp, out, mask=None, n_steps=100, lr=1e-2):
    inps = inp.view(-1, inp.shape[2])     # (B*T, d)
    outs = out.view(-1, out.shape[2])     # (B*T, D)
    B, d = inps.shape

    # Compute current approximation
    with torch.no_grad():
        x = inps @ self.truc_v.T @ self.truc_sigma  # (B, r)

    # Initialize trainable U
    updated_uT = torch.nn.Parameter(torch.matmul(torch.linalg.pinv(x), outs))  # shape: (r, D)

    optimizer = torch.optim.Adam([updated_uT], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        pred = x @ updated_uT   # (B, D)
        if mask is not None:
            loss = (((pred - outs) * mask) ** 2).sum() / (mask.sum() + 1e-6)
        else:
            loss = torch.nn.functional.mse_loss(pred, outs)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        self.updated_uT = updated_uT.detach()
        updated_output = x @ self.updated_uT
        if mask is not None:
            self.updated_error = torch.norm((updated_output - outs) * mask) / torch.norm(outs * mask)
        else:
            self.updated_error = torch.norm(updated_output - outs) / torch.norm(outs)
