import os
import pandas as pd
import spacy
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

# python3 -m spacy download en_core_web_sm
spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary:
    def __init__(self, freq_threshold):
        """
        SOS: start of sentence
        EOS: end of sentence
        UNK: unknown word
        PAD: padding
        """
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold


    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        idx = 4
        frequencies = {}

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """ returns index of that text """
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(captions_file)

        # images captions
        self.images = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.images[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)
    
class MyCollate:

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    
def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
): 
    dataset = FlickerDataset(root_folder, annotation_file, transform)

    pad_idx = dataset.vocab.stoi['<PAD>']
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
        )
    
    return loader, dataset


if __name__ == '__main__':
    transform = transforms.compose(
        [transforms.Resize((224,224), transforms.ToTensor())],
    )

    loader, dataset = get_loader(
        root_folder='flicker8k/images/',
        annotation_file='flicker8k/captions.csv',
        transform=transform,
        )
    
    for inx, (image, captions) in enumerate(loader):
        print(f'Image Shape: {image.shape} & Caption Shape: {captions.shape}')