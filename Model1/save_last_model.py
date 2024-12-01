from model import *

checkpoint = torch.load("saved_models/model_epoch_10.pth", map_location=device)
model.load_state_dict(checkpoint)

torch.save(model, "full_model.pth")
print("Full model saved as 'full_model.pth'")
