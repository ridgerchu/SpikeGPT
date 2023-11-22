from datasets import load_dataset, load_metric
from transformers import PreTrainedTokenizerFast,AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from src.model_classify import GPT, GPTConfig
import torch
import datasets
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from src.spikingjelly.clock_driven import functional
from transformers import DataCollatorWithPadding

tokenizer = PreTrainedTokenizerFast(tokenizer_file='tools/20B_tokenizer.json')

tokenizer.pad_token = "<|padding|>"
model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=12, n_embd=512, num_classes=5))
# m2 = torch.load('/share/code/SpikeGPT_addition/trained-30L-768E-936.pth',map_location=torch.device('cpu'))
# model.load_state_dict(m2, strict=False)
model = model.cuda()
def tokenize(batch):
    return tokenizer(batch["text"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset = datasets.load_from_disk('/share/datasets/sst-5')
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=16)
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=64)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=64)
def collate_fn(examples):
    examples = tokenizer.pad(
            examples,
            padding=True,
            max_length=None,
        )
    new_batch_data = []
    new_batch_label = []

    for i in range(len(examples['input_ids'])):
        new_batch_data.append(torch.tensor(examples['input_ids'][i]))
        new_batch_label.append(torch.tensor(examples['label'][i], dtype=torch.long))
    data = torch.stack(new_batch_data, dim=0)
    label = torch.stack(new_batch_label, dim=0)
    return data, label
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
head_params = list(map(id, model.classify_head.parameters()))
base_params = filter(lambda p: id(p) not in head_params,
                     model.parameters())
# optimizer = AdamW([
#             {'params': base_params},
#             {'params': model.classify_head.parameters(), 'lr': 6e-4}], lr=5e-5)
optimizer = AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Define the number of training epochs
num_epochs = 20
device = 'cuda:0'

# Loop over the epochs
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()

    # Loop over the training data in batches
    for batch in tqdm(train_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Get the inputs and labels
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        functional.reset_net(model)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Set the model to evaluation mode
    model.eval()

    # Initialize the metrics
    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Loop over the validation data in batches
    for batch in tqdm(val_loader):
        # Get the inputs and labels
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        # Update the metrics
        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += inputs.size(0)
        functional.reset_net(model)

    # Compute the metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Print the metrics
    print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")

    total_loss = 0
    total_correct = 0
    total_samples = 0

    # Loop over the validation data in batches
    for batch in tqdm(test_loader):
        # Get the inputs and labels
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        # Update the metrics
        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += inputs.size(0)
        functional.reset_net(model)

    # Compute the metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Print the metrics
    print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")

