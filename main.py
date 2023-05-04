import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
sequence_length = 1
hidden_size = 128
num_layers = 2
batch_size = 100
num_epochs = 10
learning_rate = 0.001

# Load event data
events_df = pd.read_csv('events.csv')
events_df = events_df[events_df['event'] != 'transaction']  # Filter out transactions
events_df = events_df[['visitorid', 'itemid', 'event']]  # Keep only relevant columns

# Encode visitorid and itemid
visitor_encoder = LabelEncoder()
itemid_encoder = LabelEncoder()

events_df['visitorid'] = visitor_encoder.fit_transform(events_df['visitorid'])
events_df['itemid'] = itemid_encoder.fit_transform(events_df['itemid'])

# Map event types to numbers
event_map = {'view': 0, 'addtocart': 1}
events_df['event'] = events_df['event'].map(event_map)

# Split dataset into train and test
train_df, test_df = train_test_split(events_df, test_size=0.2, random_state=42)

# Define custom dataset
class SessionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_data = torch.tensor([row['visitorid'], row['itemid']], dtype=torch.float)
        target = torch.tensor(row['event'], dtype=torch.long)
        return input_data, target

# Create datasets and dataloaders
train_dataset = SessionDataset(train_df)
test_dataset = SessionDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Update input_size and num_classes based on the dataset
input_size = 2
num_classes = len(event_map)

# Define the model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])

        return out

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

user_sequence_length = 5  # You can set this to any length you prefer

# Test the model with user input
def predict_event(visitor_item_pairs):
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor(visitor_item_pairs, dtype=torch.float).unsqueeze(0).to(device)
        output = model(input_data)
        _, prediction = torch.max(output.data, 1)
        return prediction.item()

print("\nPrediction using user input:")
print("Enter 'q' at any time to quit.\n")

while True:
    visitor_item_pairs = []

    for i in range(user_sequence_length):
        visitorid_input = input(f"Enter visitorid for interaction {i + 1}: ")
        if visitorid_input.lower() == 'q':
            break

        itemid_input = input(f"Enter itemid for interaction {i + 1}: ")
        if itemid_input.lower() == 'q':
            break

        try:
            visitorid = int(visitorid_input)
            itemid = int(itemid_input)

            # Encode visitorid and itemid using the fitted encoders
            visitorid_encoded = visitor_encoder.transform([visitorid])[0]
            itemid_encoded = itemid_encoder.transform([itemid])[0]

            visitor_item_pairs.append([visitorid_encoded, itemid_encoded])

        except ValueError:
            print("Invalid input. Please enter valid integers.")
            break
        except IndexError:
            print("Visitorid or itemid not in the original dataset. Please try with different values.")
            break

    if len(visitor_item_pairs) == user_sequence_length:
        event_prediction = predict_event(visitor_item_pairs)
        event_name = [k for k, v in event_map.items() if v == event_prediction][0]
        print(f"Predicted event for the last interaction: {event_name}")
    else:
        break

    print()
