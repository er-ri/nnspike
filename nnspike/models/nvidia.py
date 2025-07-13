import torch
import torch.nn as nn


class NvidiaModel(nn.Module):
    """
    自動運転車のEnd-to-End学習のためのNVIDIAアーキテクチャベースのニューラルネットワークモデル。

    このモデルは5つの畳み込み層の後に4つの全結合層で構成されています。最終出力層を除く各層の後にELU活性化
    関数が使用されます。さらに、間隔入力が畳み込み層からの平坦化された出力と連結されてから、全結合層に
    渡されます。

    属性:
        conv1 (nn.Conv2d): 3入力チャンネル、24出力チャンネルの第1畳み込み層
        conv2 (nn.Conv2d): 24入力チャンネル、36出力チャンネルの第2畳み込み層
        conv3 (nn.Conv2d): 36入力チャンネル、48出力チャンネルの第3畳み込み層
        conv4 (nn.Conv2d): 48入力チャンネル、64出力チャンネルの第4畳み込み層
        conv5 (nn.Conv2d): 64入力チャンネル、64出力チャンネルの第5畳み込み層
        flatten (nn.Flatten): 畳み込み層からの出力を平坦化する層
        fc1 (nn.Linear): センサー入力を含むように入力サイズが調整された第1全結合層
        fc2 (nn.Linear): 第2全結合層
        fc3 (nn.Linear): 第3全結合層
        mode_classifier (nn.Linear): 行動モード分類のための出力層（4モード）
        self_driving_head (nn.Linear): 自動運転制御のための出力層
        elu (nn.ELU): 最終出力層を除く各層の後に適用される指数線形ユニット活性化関数
        softmax (nn.Softmax): モード分類用のSoftmax活性化

    メソッド:
        forward(x, left_x, right_x, relative_position):
            モデルの順伝播を定義します。画像テンソル`x`と追加のセンサー入力を受け取り、
            ネットワークを通して処理し、2つの出力テンソルを返します：モード分類と制御。

    引数:
        x (torch.Tensor): 形状(batch_size, 3, height, width)の入力画像テンソル
        left_x (torch.Tensor): 形状(batch_size, 1)の左センサー入力テンソル
        right_x (torch.Tensor): 形状(batch_size, 1)の右センサー入力テンソル
        relative_position (torch.Tensor): 形状(batch_size, 1)の相対位置テンソル

    戻り値:
        tuple[torch.Tensor, torch.Tensor]:
            - mode_output: ロボット行動モードのSoftmax確率 (batch_size, 4)
              [左X追従, 右X追従, 障害物回避, 自動運転]
            - control_output: 自動運転モード用の制御テンソル (batch_size, 1)
    """

    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            64 * 1 * 18 + 1, 100
        )  # センサー入力（left_x、right_x、relative_position）を含めるよう入力サイズを調整
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)

        # モード分類ヘッド（4モード：左X追従、右X追従、障害物回避、自動運転）
        self.mode_classifier = nn.Linear(10, 4)

        # 自動運転制御ヘッド
        self.self_driving_head = nn.Linear(10, 1)

        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x,
        relative_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.elu(self.conv5(x))
        x = self.flatten(x)

        # 追加入力を準備
        relative_position = relative_position.view(-1, 1)

        # 平坦化された畳み込み出力と追加センサー入力を連結
        x = torch.cat([x, relative_position], dim=1)

        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))

        # モード分類出力（行動モード用softmax）
        mode_output = self.softmax(self.mode_classifier(x))

        # 自動運転制御出力
        control_output = self.self_driving_head(x)

        return mode_output, control_output
