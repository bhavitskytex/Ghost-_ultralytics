import torch 


model = torch.load('yolov11_tuned.pt', map_location='cpu',weights_only=False)
torch.save(model, 'resaved_model.pt')
