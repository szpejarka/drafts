# AlphaZero Neural Network - Copilot Instructions

## Project Overview
Educational implementation of an AlphaZero-style neural network (`alpha0.py`) that combines:
- **Policy head**: Predicts move probabilities with legal move masking support
- **Value head**: Estimates board position strength (-1 to +1 via tanh)
- **Residual architecture**: 10+ ResidualBlocks for feature extraction

## Key Architecture Patterns

### ResidualBlock Design
Located in `ResidualBlock` class: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm, with identity skip connection. Second BatchNorm uses zero-init gamma for training stability. Use when extending the residual tower.

### Network Input/Output Contract
- **Input**: `(batch, channels, height, width)` tensor
- **Output**: `(policy_logits, value)` where:
  - `policy_logits`: `(batch, action_size)` - raw logits for all possible moves
  - `value`: `(batch,)` - scalar value in range [-1, +1]
- **Legal masking**: Pass `legal_mask` `(batch, action_size)` with 1s for legal moves, 0s for illegal. Sets masked logits to -1e9.

### Initialization Strategy
- Conv2d layers: Kaiming normal (relu nonlinearity)
- Linear layers: Xavier uniform weights, zero bias
- BatchNorm: Standard (except ResidualBlock's second BN uses zero-init gamma)

## Development Workflow

### Requirements
Dependencies specified at file top: `torch`, `numpy`

### Running the Example
```bash
python alpha0.py
```
Demonstrates a small network (128 filters, 6 blocks) on random 8×8 board input.

### Testing Checklist
1. Verify output shapes match input batch size
2. Confirm legal_mask correctly zeros illegal action logits
3. Check value output is in [-1, +1] (tanh squashed)
4. Profile ResidualBlock count impact on memory/speed

## Project Conventions
- Polish variable names common (e.g., `kanały`, `planszy`, `akcji`) - preserve in docstrings
- Small-net defaults in `__main__`: 128 filters, 6 blocks (tunable for experiments)
- Action encoding: Simple from→to flattening (64×64 for 8×8 board); consider higher-order encodings if expanding

## Common Modifications
- **Change board size**: Update `board_size` param and adjust action_size accordingly
- **Tune depth**: Adjust `n_blocks` parameter (default 10)
- **Increase capacity**: Raise `filters` parameter (default 256)
- **Add regularization**: Insert dropout in policy/value heads after BN
