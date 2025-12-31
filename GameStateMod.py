from alpha0 import AlphaZeroNet
import torch

class GameState:
    """Represents a draughts game state"""
    
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = self._init_board()
        self.side_to_move = 0  # 0=player1 (white), 1=player2 (red)
        self.capture_mode = False
        self.capture_piece_idx = None
        self.history: list[(torch.Tensor,torch.Tensor)] = []
        
    def _init_board(self):
        """Initialize standard draughts starting position
        Piece encoding:
        0: empty
        1: player1 man
        2: player1 king
        3: player2 man
        4: player2 king
        """
        board = [[0] * self.board_size for _ in range(self.board_size)]
        self.history = []
        # Player 1 (white) - rows 0-2
        for r in range(3):
            for c in range(self.board_size):
                if (r + c) % 2 == 1:
                    board[r][c] = 1
        
        # Player 2 (red) - rows 5-7
        for r in range(5, self.board_size):
            for c in range(self.board_size):
                if (r + c) % 2 == 1:
                    board[r][c] = 3

        return board
    
    def to_tensor(self) -> torch.Tensor:
        """Convert board to (1, C, H, W) tensor for network input
        Channels:
        0: player1 men
        1: player1 kings
        2: player2 men
        3: player2 kings
        4: side to move (binary)
        """
        B, C, H, W = 1, 5, self.board_size, self.board_size
        x = torch.zeros(B, C, H, W, dtype=torch.float32)
        
        for r in range(H):
            for c in range(W):
                piece = self.board[r][c]
                if piece == 1:
                    x[0, 0, r, c] = 1  # player1 man
                elif piece == 2:
                    x[0, 1, r, c] = 1  # player1 king
                elif piece == 3:
                    x[0, 2, r, c] = 1  # player2 man
                elif piece == 4:
                    x[0, 3, r, c] = 1  # player2 king
        
        x[0, 4, :, :] = float(self.side_to_move)  # side to move
        
        return x
    
    def idx_to_pos(self, idx):
        """Convert flat index to (r, c)"""
        return divmod(idx, self.board_size)
    
    def pos_to_idx(self, r, c):
        """Convert (r, c) to flat index"""
        return r * self.board_size + c
    
    def encode_action(self, from_r, from_c, to_r, to_c):
        """Encode move to action_id"""
        from_idx = self.pos_to_idx(from_r, from_c)
        to_idx = self.pos_to_idx(to_r, to_c)
        return from_idx * (self.board_size * self.board_size) + to_idx
    
    def decode_action(self, action_id):
        """Decode action_id to (from_r, from_c, to_r, to_c)"""
        q = self.board_size * self.board_size
        from_idx = action_id // q
        to_idx = action_id % q
        from_r, from_c = self.idx_to_pos(from_idx)
        to_r, to_c = self.idx_to_pos(to_idx)
        return from_r, from_c, to_r, to_c
    
    def is_valid_move(self, from_r, from_c, to_r, to_c, is_capture =False):
        """Check if move is within bounds and landing square is empty"""
        if not (0 <= from_r < self.board_size and 0 <= from_c < self.board_size):
            return False
        if not (0 <= to_r < self.board_size and 0 <= to_c < self.board_size):
            return False
        if self.board[to_r][to_c] != 0:
            return False
        # white can go up, red down, kings both ways
        piece = self.board[from_r][from_c]
        if piece == 0:
            return False
        moving_owner = (piece - 1) // 2 # 0 or 1
        dr = to_r - from_r
        dc = to_c - from_c  
        if moving_owner == 0 and piece == 1 and dr <= 0 and not is_capture:
            return False
        if moving_owner == 1 and piece == 3 and dr >= 0 and not is_capture:
            return False
        # normal move or jump
        if abs(to_r - from_r) == 1 and abs(to_c - from_c) == 1:
            return True
        
        if abs(to_r - from_r) == 2 and abs(to_c - from_c) == 2:
            cap_pos = self.get_capture_target(from_r, from_c, to_r, to_c)
            if cap_pos is None:
                return False
        return True
    
    def get_capture_target(self, from_r, from_c, to_r, to_c):
        """Return captured piece position if this is a jump, else None"""
        dr = to_r - from_r
        dc = to_c - from_c

        piece = self.board[from_r][from_c]
        if piece == 0:
            return None
        moving_owner = (piece - 1) // 2

        if abs(dr) == 2 and abs(dc) == 2:
            cap_r = from_r + dr // 2
            cap_c = from_c + dc // 2
            target = self.board[cap_r][cap_c]
            if target != 0 and (target - 1) // 2 != moving_owner:
                return cap_r, cap_c
        return None
    
    def apply_action(self, action_id):  
        """Apply action to board state, return (success, is_capture)"""
        state_pre = self.to_tensor()
        success, is_capture = self.apply_action_itenral(action_id)
        if success:
            self.history.append((state_pre, action_id))
            state = self.to_tensor()

            repetition_count = sum(1 for h in self.history if torch.equal(h[0], state))
            # print(f"DEBUG: Repetition count = {repetition_count}, History size = {len(self.history)}")  # Debug

            if repetition_count > 3:
                print("Threefold repetition detected!")
                return False, False

        return success, is_capture
      
    def apply_action_itenral(self, action_id):
        """Apply action to board state, return (success, is_capture)"""
        from_r, from_c, to_r, to_c = self.decode_action(action_id)
        
        if not self.is_valid_move(from_r, from_c, to_r, to_c, self.capture_mode):
            return False, False
        
        piece = self.board[from_r][from_c]
        current_player = 0 if self.side_to_move == 0 else 1
        piece_owner = (piece - 1) // 2
        
        if piece_owner != current_player:
            return False, False
        
        # Check for capture
        cap_pos = self.get_capture_target(from_r, from_c, to_r, to_c)
        is_capture = cap_pos is not None
        
        # Move piece
        self.board[to_r][to_c] = piece
        self.board[from_r][from_c] = 0
        
        # Remove captured piece
        if is_capture:
            cap_r, cap_c = cap_pos
            self.board[cap_r][cap_c] = 0
        
        # Check promotion
        if (piece == 1 and to_r == self.board_size - 1) or \
           (piece == 3 and to_r == 0):
            self.board[to_r][to_c] = piece + 1  # Promote to king
        
        # Update state
        has_more_captures, more_captures = self._has_further_captures(to_r, to_c)
        if is_capture and has_more_captures:
            self.capture_mode = True
            self.capture_piece_idx = self.pos_to_idx(to_r, to_c)
            # display more_captures
            print("More captures available:", more_captures)
        else:
            self.capture_mode = False
            self.capture_piece_idx = None
            self.side_to_move = 1 - self.side_to_move
        
        return True, self.capture_mode
    
    def _has_further_captures(self, r, c):
        """Check if piece at (r, c) can capture again"""
        piece = self.board[r][c]
        captures = []
        if piece == 0:
            return False, []
        
        for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
            to_r, to_c = r + dr, c + dc
            if not (0 <= to_r < self.board_size and 0 <= to_c < self.board_size):
                continue
            if not self.is_valid_move(r, c, to_r, to_c, True):
                continue
            cap_pos = self.get_capture_target(r, c, to_r, to_c)
            if cap_pos is not None:
                captures.append(self.encode_action(r, c, to_r, to_c))
        if captures:
            return True,captures
        return False, []
    
    def get_legal_actions(self):
        """Return list of legal action_ids"""
        legal = []
        legal_captures = []
        if self.capture_mode:
            # Only moves from capture_piece_idx
            r, c = self.idx_to_pos(self.capture_piece_idx)

            has_captures,captures = self._has_further_captures(r, c)  
            if has_captures:  
                legal_captures.extend(captures)
            # All jumps from this piece
            # piece = self.board[r][c]
            # for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
            #     to_r, to_c = r + dr, c + dc
            #     if self.is_valid_move(r, c, to_r, to_c):
            #         if self.get_capture_target(r, c, to_r, to_c):
            #             legal.append(self.encode_action(r, c, to_r, to_c))
        else:
            # All moves for current player
            for r in range(self.board_size):
                for c in range(self.board_size):
                    piece = self.board[r][c]
                    if piece == 0:
                        continue
                    piece_owner = (piece - 1) // 2
                    if piece_owner != self.side_to_move:
                        continue
                    # Try all moves
                    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 2), (2, -2), (-2, 2), (-2, -2)]:
                        to_r, to_c = r + dr, c + dc
                        if self.is_valid_move(r, c, to_r, to_c):
                            if abs(dr) == 2 and abs(dc) == 2:
                                legal_captures.append(self.encode_action(r, c, to_r, to_c))
                            else:
                                legal.append(self.encode_action(r, c, to_r, to_c))
        if legal_captures:
            return legal_captures
        return legal
    
    def get_legal_mask(self):
        """Return legal_mask tensor (1, 4096) for network"""
        action_size = self.board_size * self.board_size * self.board_size * self.board_size
        mask = torch.zeros(1, action_size, dtype=torch.uint8)
        legal = self.get_legal_actions()
        if legal:
            mask[0, torch.tensor(legal, dtype=torch.long)] = 1
        return mask
    
    def clone(self):
        """Create a copy of this state"""
        new_state = GameState(self.board_size)
        new_state.board = [row[:] for row in self.board]
        new_state.side_to_move = self.side_to_move
        new_state.capture_mode = self.capture_mode
        new_state.capture_piece_idx = self.capture_piece_idx
        return new_state
    
    def is_terminal(self):
        """Check if game is over"""
        return len(self.get_legal_actions()) == 0
    
    def get_result(self):
        """Return game result: +1 if current player winning, -1 if losing, 0 draw"""
        if self.is_terminal():
            return -1  # Current player lost (no legal moves)
        return 0  # Game ongoing
    
    def display(self):
        """Display the board state in a readable format"""
        print("\n   0 1 2 3 4 5 6 7")
        print("  -───────────────")
        
        for r in range(self.board_size):
            row_str = f"{r}│"
            for c in range(self.board_size):
                piece = self.board[r][c]
                
                # Piece symbols
                if piece == 0:
                    if (r + c) % 2 == 1:
                        row_str += " ·"  # playable square
                    else:
                        row_str += " ·"  # non-playable (visual)
                elif piece == 1:
                    row_str += " ●"  # player1 man
                elif piece == 2:
                    row_str += " ◉"  # player1 king
                elif piece == 3:
                    row_str += " ○"  # player2 man
                elif piece == 4:
                    row_str += " ◎"  # player2 king
            
            row_str += f"│{r}"
            print(row_str)
        
        print("  -───────────────")
        print("   0 1 2 3 4 5 6 7\n")
        
        # Game state info
        player = "White (●)" if self.side_to_move == 0 else "Red (○)"
        print(f"Side to move: {player}")
        
        if self.capture_mode:
            cap_r, cap_c = self.idx_to_pos(self.capture_piece_idx)
            print(f"Capture mode: piece at ({cap_r}, {cap_c}) must continue capturing")
        
        legal = self.get_legal_actions()
        print(f"Legal moves: {len(legal)}\n")   

# In __main__ or your test:
if __name__ == "__main__":
    state = GameState(board_size=8)

    net = AlphaZeroNet(board_channels=5, board_size=8, action_size=64*64,
                       filters=128, n_blocks=6)
    net.load_model("trained_model.pth")  # Load pre-trained model if available
    state.display()

    while not state.is_terminal():

        x = state.to_tensor().to(net.device)
        legal_mask = state.get_legal_mask().to(net.device)
        
        logits, value = net(x, legal_mask=legal_mask)
        probs = torch.softmax(logits[0], dim=0)
        
        legal = state.get_legal_actions()
        action = legal[torch.argmax(probs[legal])]
        
        success, is_capture = state.apply_action(action)
        if not success:
            print("Error: invalid move attempted by AI")
            break
        state.display()
        print(f"Move applied: success={success}, capture={is_capture}")
        if state.is_terminal():
            print("Game over!")
            result = state.get_result()
            if result == -1 and state.side_to_move == 1:
                print("White player wins!")
            elif result == -1 and state.side_to_move == 0:
                print("Red player wins!")
            else:
                print("It's a draw!")
            break   
        readline = input("Press Enter to continue, 'q' to quit: ")
        if readline.lower() == 'q':
            break
