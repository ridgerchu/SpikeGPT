from datasets import load_dataset, load_metric
from transformers import PreTrainedTokenizerFast,AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset
from src.model_classify import GPT, GPTConfig
import torch
import datasets
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW
from src.spikingjelly.clock_driven import functional
from transformers import DataCollatorWithPadding
import os

tokenizer = PreTrainedTokenizerFast(tokenizer_file='tools/20B_tokenizer.json')

tokenizer.pad_token = "<|padding|>"
model = GPT(GPTConfig(vocab_size=50277, ctx_len=1024, model_type='RWKV', n_layer=24, n_embd=768))
#m2 = torch.load('/share/code/SpikeGPT_addition/trained-30L-768E-936.pth',map_location=torch.device('cpu'))
#model.load_state_dict(m2, strict=False)
model = model.cuda()
def tokenize(batch):
    return tokenizer(batch["sentence"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# dataset = load_dataset("glue", "sst2", keep_in_memory=True, cache_dir="/share/datasets")
dataset = datasets.load_from_disk('/share/datasets/sst-2')
#dataset.save_to_disk('/share/datasets/subj')
train_dataset = dataset["train"]
test_dataset = dataset["validation"]
eval_dataset = dataset["test"]
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=16)
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=64)
eval_dataset = eval_dataset.map(tokenize, batched=True, batch_size=64)
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
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
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
