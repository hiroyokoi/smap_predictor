import torch
from commons import get_model, transform_image

model = get_model()

# Obtain class list
model_classes = ['Goro_Inagaki', 'Masahiro_Nakai', 'Shingo_Katori', 'Takuya_Kimura', 'Tsuyoshi_Kusanagi']
model_classes = {idx: each for idx, each in enumerate(model_classes)}

# Predict image
def get_prediction(image_bytes, topk=5):
    try:
        tensor = transform_image(image_bytes)
        out = model.forward(tensor)
    except Exception:
        return 0, 'error'
    ps = torch.exp(out)
    # Find the topk predictions
    topk, topclass = ps.topk(topk, dim=1)
    topk, topclass = topk.squeeze().detach().numpy(), topclass.squeeze().detach().numpy()
    # Extract the actual classes and probabilities
    prob_dict = {}
    for i in range(len(model_classes)):
      prob_dict[model_classes[topclass[i]]] = topk[i]
    return prob_dict