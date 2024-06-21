import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 356

loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    
    return image.to(device)

class CustomVGG(nn.Module):
    def __init__(self):
        super(CustomVGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29]
        
    def forward(self,x):
        features = []
        
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            
            if str(layer_num) in self.chosen_features:
                features.append(x)
                
        return features

model = CustomVGG().to(device).eval()
original_img = load_image("annahathaway.png")
style_img = load_image("style.jpg")
generated = original_img.clone().requires_grad_(True)
# Hyperparams
total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_img_features = model(style_img)
    
    style_loss = org_loss = 0
    
    for gen_feature, org_feature, style_feature in zip(generated_features, original_img_features, style_img_features):
        batch_size, channel, height, width = gen_feature.shape
        org_loss += torch.mean((gen_feature - org_feature) ** 2)
        
        # Compute Gram Matrix
        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )
        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
        )
        
        style_loss += torch.mean((G-A)**2)
        
    total_loss = alpha*org_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if step%200 == 0:
        print(total_loss)
        save_image(generated, f'generated/img{int(step/200)}.png')
