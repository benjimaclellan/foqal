import torch
from foqal.causal.classical import ClassicalCommonCause


if __name__ == "__main__":
    print(f"CUDA is available: {torch.cuda.is_available()}")
    device = "cuda"
    model = ClassicalCommonCause(n_settings=10, latent_dim=10).to(device)

    pred = model.forward()
    model.initialize_params()
    model.to(device)
    print(pred.device)
    data = model.forward()
    loss = torch.nn.MSELoss()
    l = loss(pred, data)
    print(l)
