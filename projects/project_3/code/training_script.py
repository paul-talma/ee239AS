import torch
from train import solver

path_to_gpt_tester = "./pretrained_models/minigpt_tester.pt"  # Load the gpt model with name minigpt_tester.pt

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)


solver(model_name="bigram")
