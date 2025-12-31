import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from alpha0 import AlphaZeroNet

class GameDataset(Dataset):
    """Dataset for loading saved game data"""
    
    def __init__(self, data_dir="training_data"):
        """Load all .pt files from data directory"""
        self.data_dir = Path(data_dir)
        self.states = []
        self.actions = []
        self.values = []
        
        # Load all game files
        for game_file in self.data_dir.glob("*.pt"):
            data = torch.load(game_file)
            self.states.append(data['states'])
            self.actions.append(data['actions'])
            self.values.append(data['values'])
        
        # Concatenate all games
        self.states = torch.cat(self.states, dim=0)
        self.actions = torch.cat(self.actions, dim=0)
        self.values = torch.cat(self.values, dim=0)
        
        print(f"Loaded {len(self.states)} training examples from {len(list(self.data_dir.glob('*.pt')))} games")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.values[idx]


def train_supervised(net, data_dir="training_data", epochs=10, batch_size=32, lr=0.001):
    """
    Train network using supervised learning from saved games
    
    Args:
        net: AlphaZeroNet instance
        data_dir: Directory with saved game files
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    # Setup    
    # Load dataset
    dataset = GameDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        net.train()
        for states, actions, values in dataloader:
            states = states.to(net.device)
            actions = actions.to(net.device)
            values = values.to(net.device)
            
            # Forward pass
            policy_logits, value_pred = net(states)
            
            # Policy loss (cross-entropy: learn to predict the action taken)
            policy_loss = F.cross_entropy(policy_logits, actions)
            
            # Value loss (MSE: learn to predict game outcome)
            value_loss = F.mse_loss(value_pred.squeeze(), values)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        # Epoch summary
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = total_loss / num_batches
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={avg_total_loss:.4f}, "
              f"Policy={avg_policy_loss:.4f}, "
              f"Value={avg_value_loss:.4f}")
    
    return net


def evaluate_network(net, data_dir="training_data"):
    """Evaluate network accuracy on saved games"""
    
    dataset = GameDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    correct_actions = 0
    total_actions = 0
    total_value_error = 0.0
    
    with torch.no_grad():
        for states, actions, values in dataloader:
            states = states.to(net.device)
            actions = actions.to(net.device)
            values = values.to(net.device)
            
            policy_logits, value_pred = net(states)
            
            # Policy accuracy
            predicted_actions = torch.argmax(policy_logits, dim=1)
            correct_actions += (predicted_actions == actions).sum().item()
            total_actions += len(actions)
            
            # Value error
            value_error = torch.abs(value_pred.squeeze() - values).sum().item()
            total_value_error += value_error
    
    policy_accuracy = correct_actions / total_actions * 100
    avg_value_error = total_value_error / total_actions
    
    print(f"\nEvaluation Results:")
    print(f"Policy Accuracy: {policy_accuracy:.2f}%")
    print(f"Avg Value Error: {avg_value_error:.4f}")
    
    return policy_accuracy, avg_value_error


# Example usage
if __name__ == "__main__":
    # Create network
    net = AlphaZeroNet(board_channels=5, board_size=8, action_size=64*64,
                       filters=128, n_blocks=6)
    net.load_model("trained_model.pth")  # Load pre-trained model if available
    print("Model loaded from trained_model.pth.")
    # Train on saved games
    print("Training network on saved games...")
    train_supervised(net, data_dir="training_data", epochs=5, batch_size=32, lr=0.001)
    
    # Evaluate
    print("\nEvaluating network...")
    evaluate_network(net, data_dir="training_data")
    
    # Save trained model
    net.save_model("trained_model.pth")
    print("\nModel saved to trained_model.pth")