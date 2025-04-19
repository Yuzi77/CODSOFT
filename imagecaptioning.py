import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configuration class
class Config:
    encoder_type = "resnet152"  # Options: "resnet152", "vgg19"
    decoder_type = "lstm"       # Options: "lstm", "transformer"
    embed_size = 256
    hidden_size = 512
    attention_dim = 512
    num_layers = 2
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary class with proper indexing methods
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.idx = 4
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)

# CNN Encoder: Supports both ResNet and VGG
class Encoder(nn.Module):
    def __init__(self, encoder_type, embed_size):
        super(Encoder, self).__init__()
        
        if encoder_type.startswith("resnet"):
            # ResNet encoder
            cnn = getattr(models, encoder_type)(pretrained=True)
            modules = list(cnn.children())[:-2]  # Remove final FC layers
            self.cnn = nn.Sequential(*modules)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, embed_size)
        
        elif encoder_type.startswith("vgg"):
            # VGG encoder
            cnn = getattr(models, encoder_type)(pretrained=True)
            self.cnn = cnn.features
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, embed_size)
        
        # Freeze CNN parameters
        for param in self.cnn.parameters():
            param.requires_grad = False
    
    def forward(self, images):
        features = self.cnn(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(TransformerDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, 100)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, nhead=8, dim_feedforward=hidden_size
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, features, captions):
        # Create masks
        tgt_mask = generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        
        # Embed and add positional encoding
        tgt = self.embed(captions) * np.sqrt(self.embed.embedding_dim)
        tgt = self.positional_encoding(tgt)
        
        # Reshape for transformer: [seq_len, batch_size, embed_dim]
        tgt = tgt.permute(1, 0, 2)
        
        # Create memory from features
        memory = features.unsqueeze(0)
        
        # Forward through transformer
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = output.permute(1, 0, 2)
        
        return self.fc_out(output)

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Helper function for transformer
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Complete Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, config, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        
        # Create encoder
        self.encoder = Encoder(config.encoder_type, config.embed_size)
        
        # Create appropriate decoder
        if config.decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                config.embed_size, config.hidden_size, 
                vocab_size, config.num_layers
            )
        else:
            self.decoder = TransformerDecoder(
                config.embed_size, config.hidden_size, 
                vocab_size, config.num_layers
            )
        
        self.decoder_type = config.decoder_type
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

# Image preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to generate captions (Fixed for the error in the screenshot)
def generate_caption(model, image_path, vocab, device):
    # Prepare image
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Set model to eval mode
    model.eval()
    
    with torch.no_grad():
        # Get features
        features = model.encoder(image_tensor)
        
        # Start with <start> token - FIX: Use vocab.word2idx instead of vocab directly
        caption = [vocab.word2idx["<start>"]]
        
        # Generate each word of caption
        for _ in range(20):  # max length
            caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(device)
            
            if model.decoder_type == "lstm":
                outputs = model.decoder(features, caption_tensor)
                predicted = outputs[:, -1].argmax(dim=1).item()
            else:
                outputs = model.decoder(features, caption_tensor)
                predicted = outputs[:, -1].argmax(dim=1).item()
            
            caption.append(predicted)
            
            if predicted == vocab.word2idx["<end>"]:
                break
        
        # Convert indices to words - FIX: Use vocab.idx2word instead of vocab.word2idx
        words = [vocab.idx2word[idx] for idx in caption[1:-1]]  # Remove <start> and <end>
        
    return " ".join(words)

# Display captioned image
def display_captioned_image(image_path, caption, output_path=None):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    width, height = image.size
    
    # Try to load a nice font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Calculate text size and position
    try:
        text_width, text_height = draw.textbbox((0, 0), caption, font=font)[2:]
    except:
        text_width, text_height = draw.textsize(caption, font=font)
    
    text_x = (width - text_width) // 2
    text_y = height - text_height - 10
    
    # Draw text background
    draw.rectangle([(0, text_y - 5), (width, text_y + text_height + 5)], fill=(0, 0, 0, 128))
    
    # Draw text
    draw.text((text_x, text_y), caption, font=font, fill=(255, 255, 255))
    
    if output_path:
        image.save(output_path)
    else:
        image.show()
    
    return image

# Filter to show only activities
def filter_activities(caption):
    activity_words = ["sitting", "running", "walking", "jumping", "eating", 
                     "sleeping", "playing", "standing", "sits", "runs", 
                     "walks", "jumps", "eats", "sleeps", "plays", "stands"]
    
    words = caption.split()
    activities = [word for word in words if word.lower() in activity_words]
    
    if activities:
        return " ".join(activities)
    return caption

# Pretrained model option (no dataset needed)
def caption_with_pretrained(image_path, output_path=None, show_activities_only=False):
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        # Load model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Process image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        
        # Generate caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Filter for activities if needed
        if show_activities_only:
            caption = filter_activities(caption)
        
        # Display captioned image
        display_captioned_image(image_path, caption, output_path)
        
        return caption
        
    except ImportError:
        print("Please install transformers library: pip install transformers")
        return None

# Example usage with BLIP pretrained model (easiest option)
def main_pretrained():
    # Replace with your image path
    image_path = "example.png"
    output_path = "captioned_image.png"
    
    # Generate caption
    caption = caption_with_pretrained(
        image_path, 
        output_path, 
        show_activities_only=True
    )
    
    print(f"Generated Caption: {caption}")

# Example usage with custom model (requires training)
def main_custom():
    # Create vocabulary
    vocab = Vocabulary()
    for word in ["cat", "dog", "sits", "runs", "walks", "jumps", "on", "the", "mat"]:
        vocab.add_word(word)
    
    # Create config
    config = Config()
    
    # Create model
    model = ImageCaptioningModel(config, len(vocab)).to(config.device)
    
    # Generate caption
    image_path = "example.png"
    caption = generate_caption(model, image_path, vocab, config.device)
    
    # Filter for activities only
    activity_caption = filter_activities(caption)
    
    # Display captioned image
    display_captioned_image(image_path, activity_caption, "captioned_image.png")
    
    print(f"Generated Caption: {caption}")
    print(f"Activities: {activity_caption}")

if __name__ == "__main__":
    # Use pretrained option (recommended, no dataset needed)
    main_pretrained()
    
    # OR use custom model (requires training)
    # main_custom()
