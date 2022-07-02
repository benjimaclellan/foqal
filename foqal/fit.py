import tqdm
from torch.functional import F


def fit(model, data, optimizer, n_steps=1000):
    """
    Training loop for torch model.
    """

    losses = []
    for step in (pbar := tqdm.tqdm(range(n_steps))):

        pred = model.forward()
        loss = F.mse_loss(pred, data)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if loss.is_cuda:
            losses.append(loss.cpu().detach().numpy())
        else:
            losses.append(loss.detach().numpy())

        pbar.set_description(f"Cost: {losses[-1]:.4f}")

    return losses