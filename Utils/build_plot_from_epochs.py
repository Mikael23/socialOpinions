import pandas as pd
import matplotlib.pyplot as plt
import json
import plotly.express as px

# Load your JSON log
with open('/Users/I552581/Library/Application Support/JetBrains/PyCharmCE2024.3/scratches/100_epochs.json') as f:
    logs = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(logs)

# Aggregate per epoch
# There can be multiple rows per epoch (loss and eval_loss)
grouped = df.groupby('epoch').agg({
    'loss': 'first',
    'eval_loss': 'first',
    'train_loss': 'first'
}).reset_index()

# Round to 3 decimals
grouped['loss'] = grouped['loss'].round(3)
grouped['eval_loss'] = grouped['eval_loss'].round(3)
grouped['train_loss'] = grouped['train_loss'].round(3)

print(grouped)

# Sample data

# Create the scatter plot
fig = px.line(
    data_frame=grouped,
    x=grouped['epoch'],
    y=[grouped['loss'], grouped['eval_loss'], grouped['train_loss']],
    labels={'epoch': 'Epoch', 'loss': 'Loss'},
    title="Training and Valid Loss per Epoch")
# Display the plot
fig.show()
