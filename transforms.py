
from torchvision import transforms as transforms

img_train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomRotation((-15,15)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])