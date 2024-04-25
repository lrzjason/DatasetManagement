import torch
from PIL import Image
from torchvision import transforms
from metaformer import caformer

# load the image and preprocess it
image = Image.open('dog.jpg')
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
image = transform(image).unsqueeze(0) # add batch dimension

# load the model and the labels
model = caformer.CAFormer_B36(pretrained=True) # load the CAFormer-B36 model
model.eval() # set the model to evaluation mode
labels = torch.load('imagenet_labels.pt') # load the ImageNet labels

# make a prediction and get the top 5 classes
with torch.no_grad():
    output = model(image) # forward pass
    prob = torch.nn.functional.softmax(output, dim=1) # convert logits to probabilities
    top5_prob, top5_id = torch.topk(prob, 5) # get the top 5 probabilities and indices

# print the results
print('Top 5 predictions:')
for i in range(5):
    print(f'{i+1}. {labels[top5_id[0][i]]}: {top5_prob[0][i].item():.4f}')
