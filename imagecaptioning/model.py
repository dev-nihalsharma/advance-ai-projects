import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True)
        self.inception.aux_logits = False

        for parameter in self.inception.parameters():
            parameter.requires_grad = False
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)

        # for name, param in self.inception.named_parameters():
        #     if 'fc.weight' in name or 'fc.bias' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        caption_embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), caption_embeddings), dim=0)
        hiddens,_ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs
    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size, train_CNN=False)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)

        return outputs
    def caption_image(self, image, vocabulary, max_length=50):
        result_captions = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            state = None
            for _ in range(max_length):
                hiddens, state = self.decoder.lstm(x, state)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_captions.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)


                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break

        return [vocabulary.itos[idx] for idx in result_captions ]