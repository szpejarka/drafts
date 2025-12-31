import torch
import random
from pathlib import Path
from datetime import datetime
from GameStateMod import GameState

class PlayGame:
    """Play one game and save history for training"""
    
    def __init__(self, net, board_size=8, temperature=1.0, save_dir="game_data"):
        """
        Args:
            net: AlphaZeroNet instance
            board_size: Board size (default 8x8)
            temperature: Sampling temperature for move selection (1.0=stochastic, 0.0=greedy)
            save_dir: Directory to save game history
        """
        self.net = net
        self.board_size = board_size
        self.temperature = temperature
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training data: (state, policy, value)
        self.training_examples = []
        
    def select_action(self, state, deterministic=False):
        """Select action using network policy with optional temperature"""
        x = state.to_tensor().to(self.net.device)  # (1, C, H, W)
        legal_mask = state.get_legal_mask().to(self.net.device)  # (1, action_size)
        
        with torch.no_grad():
            logits, value = self.net(x, legal_mask=legal_mask)
            
        # Get probabilities
        probs = torch.softmax(logits[0] / self.temperature, dim=0)
        legal_actions = state.get_legal_actions()
        
        if deterministic or self.temperature < 0.01:
            # Greedy: pick best legal action
            action = legal_actions[torch.argmax(probs[legal_actions])]
        else:
            # Sample from distribution
            legal_probs = probs[legal_actions]
            legal_probs = legal_probs / legal_probs.sum()  # Renormalize
            action_idx = torch.multinomial(legal_probs, 1).item()
            action = legal_actions[action_idx]
        
        return action, probs, value.item()
    
    def play_game(self, max_moves=200, verbose=False):
        """
        Play one full game
        
        Returns:
            game_result: 1 if player 0 wins, -1 if player 1 wins, 0 for draw
            num_moves: Number of moves played
        """
        state = GameState(self.board_size)
        move_count = 0
        
        if verbose:
            state.display()
        
        while not state.is_terminal() and move_count < max_moves:
            # Get current state representation
            state_tensor = state.to_tensor()
            current_player = state.side_to_move
            
            # Select action
            action, policy, value = self.select_action(state)
            
            # Store training example (state, policy, placeholder for value)
            # Value will be filled in after game ends
            self.training_examples.append({
                'state': state_tensor.clone(),
                'action': action,
                'policy': policy.clone(),
                'player': current_player,
                'value': None  # To be filled
            })
            
            # Apply action
            success, is_capture = state.apply_action(action)
            
            if not success:
                print(f"ERROR: Invalid move on turn {move_count}")
                return None, move_count
            
            move_count += 1
            
            if verbose:
                from_r, from_c, to_r, to_c = state.decode_action(action)
                print(f"\nMove {move_count}: ({from_r},{from_c}) -> ({to_r},{to_c})")
                print(f"Capture mode: {is_capture}, Value estimate: {value:.3f}")
                state.display()
        
        # Determine game result
        if state.is_terminal():
            # Current player lost (no legal moves)
            winner = 1 - state.side_to_move
            if verbose:
                print(f"\nGame Over! Player {winner} wins!")
        else:
            # Max moves reached - draw
            winner = None
            if verbose:
                print(f"\nDraw! Max moves ({max_moves}) reached")
        
        # Fill in final values for all training examples
        self._assign_values(winner)
        
        return winner, move_count
    
    def _assign_values(self, winner):
        """Assign final game result to all training examples"""
        for example in self.training_examples:
            if winner is None:
                # Draw
                example['value'] = 0.0
            elif example['player'] == winner:
                # This player won
                example['value'] = 1.0
            else:
                # This player lost
                example['value'] = -1.0
    
    def save_game(self, filename=None):
        """Save training examples to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{timestamp}.pt"
        
        filepath = self.save_dir / filename
        
        # Convert to tensor format
        states = torch.cat([ex['state'] for ex in self.training_examples], dim=0)
        policies = torch.stack([ex['policy'] for ex in self.training_examples])
        values = torch.tensor([ex['value'] for ex in self.training_examples], dtype=torch.float32)
        actions = torch.tensor([ex['action'] for ex in self.training_examples], dtype=torch.int64)
        
        # Save
        torch.save({
            'states': states,
            'actions': actions,
            'policies': policies,
            'values': values,
            'num_examples': len(self.training_examples)
        }, filepath)
        
        print(f"Saved {len(self.training_examples)} training examples to {filepath}")
        return filepath
    
    def reset(self):
        """Clear training examples for next game"""
        self.training_examples = []


# Example usage
if __name__ == "__main__":
    from alpha0 import AlphaZeroNet
    
    # Create network
    net = AlphaZeroNet(board_channels=5, board_size=8, action_size=64*64,
                       filters=128, n_blocks=6)
    net.load_model("trained_model.pth")  # Load pre-trained model if available
    # Play one game
    game = PlayGame(net, temperature=1.0, save_dir="training_data")
    # winner, num_moves = game.play_game(verbose=True)
    
    # if winner is not None:
    #     print(f"\nGame finished in {num_moves} moves. Winner: Player {winner}")
    # else:
    #     print(f"\nGame ended in draw after {num_moves} moves")
    
    # # Save training data
    # game.save_game()
    
    # Play multiple games
    print("\n" + "="*50)
    print("Playing 5 games...")
    print("="*50)
    
    for i in range(50):
        game.reset()
        winner, moves = game.play_game(verbose=False)
        if winner is not None:
            game.save_game(f"game_{i:03d}.pt")
            print(f"Game {i+1}: {moves} moves, Winner: {winner}")