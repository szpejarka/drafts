# requirements: torch, numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # zero-init second BN gamma can help (stabilize training)
        nn.init.constant_(self.bn2.weight, 0.0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroNet(nn.Module):
    def __init__(self, board_channels, board_size, action_size,
                 filters=256, n_blocks=10):
        """
        board_channels: liczba kanałów wejścia (np. 5)
        board_size: rozmiar planszy (8 for 8x8)
        action_size: liczba wszystkich możliwych akcji (np. 64*64)
        filters: liczba filtrów w convach
        n_blocks: liczba residual blocks
        """
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size

        # wejściowy stem
        self.conv_stem = nn.Conv2d(board_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(filters)

        # residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(filters) for _ in range(n_blocks)])

        # policy head: conv 1x1 -> BN -> ReLU -> fc -> logits for actions
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # value head: conv 1x1 -> BN -> ReLU -> fc -> tanh
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.todevice()

    def forward(self, x, legal_mask=None):
        """
        x: tensor (B, C, H, W)
        legal_mask: optional mask (B, action_size) with 1 for legal actions, 0 for illegal
        returns: (policy_logits, value)
        """
        out = self.conv_stem(x)
        out = self.bn_stem(out)
        out = F.relu(out)

        for b in self.res_blocks:
            out = b(out)

        # policy head
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)              # (B, 2*H*W)
        logits = self.policy_fc(p)             # (B, action_size)

        # mask illegal moves by setting logits to large negative
        if legal_mask is not None:
            # legal_mask: 1 for legal, 0 for illegal
            neg_inf = -1e9
            logits = logits.masked_fill(legal_mask == 0, neg_inf)

        # value head
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)              # (B, H*W)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B,)

        return logits, v
    def save_model(self, filepath):
        """Save model weights to file"""
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """Load saved model weights"""
        self.load_state_dict(torch.load(filepath))
        self.eval()

    def todevice(self):
        """Move model to specified device"""
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")   
        return super().to(self.device)
# przykładowe użycie:
if __name__ == "__main__":
    B = 4
    C = 5
    H = W = 8
    print(torch.__version__)
    print(torch.backends.mps.is_available()) # Should return True
    ACTIONS = 64*64  # prosty encoding from->to (można ulepszyć)
    net = AlphaZeroNet(board_channels=C, board_size=H, action_size=ACTIONS,
                       filters=128, n_blocks=6)  # small net
    sample_in = torch.randn(B, C, H, W)
    # legal mask random example:
    legal_mask = torch.zeros(B, ACTIONS, dtype=torch.uint8)
    legal_mask[:, :20] = 1
    logits, value = net(sample_in, legal_mask=legal_mask)
    print("logits", logits.shape, "value", value.shape)