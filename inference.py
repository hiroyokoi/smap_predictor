import torch
from commons import get_model, transform_image

model = get_model()

# Obtain class list
model_classes = ['Goro_Inagaki', 'Masahiro_Nakai', 'Shingo_Katori', 'Takuya_Kimura', 'Tsuyoshi_Kusanagi']
model_classes = {str(idx): each for idx, each in enumerate(model_classes)}
model_classes['1']

# Predict image
def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes)
        out = model.forward(tensor)
    except Exception:
        return 0, 'error'
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return model_classes[predicted_idx]