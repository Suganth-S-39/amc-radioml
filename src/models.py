"""Model architectures for automatic modulation classification."""
import torch
import torch.nn as nn
class CNN_BiLSTM_Attention(nn.Module):
    """ Baseline: single-scale CNN followed by BiLSTM and additive attention.
        88,716 trainable parameters.
    """

    def __init__(self, num_classes=11):
        super().__init__()
        self.conv1=nn.Conv1d(2, 64, 3, padding=1)
        self.bn1 =nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.bn2= nn.BatchNorm1d(64)
        self.pool= nn.MaxPool1d(2)
        self.relu= nn.ReLU()
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.pool(self.relu(self.bn1(self.conv1(x))))
        out = self.pool(self.relu(self.bn2(self.conv2(out))))
        out = out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)
        w = torch.softmax(self.attn_fc(lstm_out), dim=1)
        ctx = torch.sum(w * lstm_out, dim=1)
        out = self.dropout(self.relu(self.fc1(ctx)))
        return self.fc2(out), None


class MultiScaleCNN_BiLSTM_Attention(nn.Module):
    """ Proposed model: parallel conv branches (k=3,5,7), BiLSTM, attention.
        95,532 trainable parameters.
    """

    def __init__(self, num_classes=11):
        super().__init__()
        self.branch3 = nn.Sequential(nn.Conv1d(2, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())
        self.branch5 = nn.Sequential(nn.Conv1d(2, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU())
        self.branch7 = nn.Sequential(nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU())

        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(96, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        out = self.pool(out)
        out = self.pool(self.relu(self.bn2(self.conv2(out))))
        out = out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)
        w = torch.softmax(self.attn_fc(lstm_out), dim=1)
        ctx = torch.sum(w * lstm_out, dim=1)
        out = self.dropout(self.relu(self.fc1(ctx)))
        return self.fc2(out), None


class SNRAwareFiLMModel(nn.Module):
    """ Ablation: blind SNR estimator conditioning BiLSTM features via FiLM.
        Reported as a negative result — provides no benefit once power
        normalization is applied.
    """

    def __init__(self, num_classes=11):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)

        # blind SNR estimator — GroupNorm avoids train/eval statistics mismatch
        self.snr_conv1 = nn.Conv1d(2, 32, 3, padding=1)
        self.snr_gn1 = nn.GroupNorm(8, 32)
        self.snr_conv2 = nn.Conv1d(32, 32, 3, padding=1)
        self.snr_gn2 = nn.GroupNorm(8, 32)
        self.snr_pool = nn.AdaptiveAvgPool1d(1)
        self.snr_fc = nn.Linear(32, 1)

        # FiLM generator: SNR scalar -> per-channel scale and shift
        self.film = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 256))
        self.attn_fc = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        s = self.relu(self.snr_gn1(self.snr_conv1(x)))
        s = self.relu(self.snr_gn2(self.snr_conv2(s)))
        snr_est = torch.tanh(self.snr_fc(self.snr_pool(s).squeeze(-1)))

        out = self.pool(self.relu(self.bn1(self.conv1(x))))
        out = self.pool(self.relu(self.bn2(self.conv2(out))))
        out = out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)

        gamma, beta = self.film(snr_est).chunk(2, dim=-1)
        mod = (1 + gamma.unsqueeze(1)) * lstm_out + beta.unsqueeze(1)

        w = torch.softmax(self.attn_fc(mod), dim=1)
        ctx = torch.sum(w * mod, dim=1)
        out = self.dropout(self.relu(self.fc1(ctx)))
        return self.fc2(out), snr_est


MODELS = {
    'baseline': CNN_BiLSTM_Attention,
    'multiscale': MultiScaleCNN_BiLSTM_Attention,
    'film': SNRAwareFiLMModel,
}


if __name__ == '__main__':
    for name, cls in MODELS.items():
        n = sum(p.numel() for p in cls().parameters())
        print(f"{name:12s}: {n:,} parameters")