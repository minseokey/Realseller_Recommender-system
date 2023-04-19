import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, hidden_dim):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.hidden_dim = hidden_dim

        # Conv 입니다, 논문에서 3,4,5 하라고 해서 반복문으로 Conv 를 3개 만들어 줍니다.
        self.conv_layers = nn.ModuleList()
        for filter_size in filter_sizes:
            self.conv_layers.append(nn.Conv2d(1, num_filters, (filter_size, embedding_dim)))

        # Max-pooling 입니다
        self.maxpool = nn.MaxPool2d((64, 1))

        # Fully connected 입니다. linear 로 펴서 비교합니다.
        self.fc = nn.Linear(num_filters * len(filter_sizes), hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Output
        self.out = nn.Linear(hidden_dim, 1)


    def forward_once(self, x):
        x = x.unsqueeze(1)
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = nn.functional.relu(conv_layer(x))
            conv_output = conv_output.squeeze(3)
            pool_output = nn.functional.max_pool1d(conv_output, conv_output.size(2))
            conv_outputs.append(pool_output)
        x = torch.cat(conv_outputs, 1)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, input1, input2, target):
        self.model.train()
        self.optimizer.zero_grad()
        output1, output2 = self.model(input1, input2)
        target = target.view(-1, 1)
        loss = self.loss_fn((output1 - output2)**2, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def test_step(self, input1, input2):
        self.model.eval()
        with torch.no_grad():
            output1, output2 = self.model(input1, input2)
            dist = torch.abs(output1 - output2)

        return dist.cpu().numpy()


# 하이퍼파리미터
EMBEDDING_DIM = 300
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
HIDDEN_DIM = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 50
NUM_EPOCHS = 25

# 모델 정의, 하이퍼 파라미터 위에 지정해두었습니다.
model = SiameseNetwork(
    embedding_dim=EMBEDDING_DIM,
    num_filters=NUM_FILTERS,
    filter_sizes=FILTER_SIZES,
    hidden_dim=HIDDEN_DIM
)

# loss 와 optimizer 정의.
loss_fn = nn.HingeEmbeddingLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 만들어 놓은 정보들로 트레이너 설정
trainer = SiameseTrainer(model, loss_fn, optimizer)

# 더미데이터를 만들어서 했습니다. 그냥 데이터로 하면 계속 같은값으로 나와서...
x1 = torch.randn((BATCH_SIZE, 64, EMBEDDING_DIM))
x2 = torch.randn((BATCH_SIZE, 64, EMBEDDING_DIM))
y = torch.randint(0, 2, (BATCH_SIZE,))

# 에폭이나 배치사이즈 같은것들 다 하이퍼 파리미터 자리로 뻈습니다.
show = []
for epoch in range(NUM_EPOCHS):
    avg_loss = 0.0
    for i in range(0, len(x1), BATCH_SIZE):
        batch_x1 = x1[i:i + BATCH_SIZE]
        batch_x2 = x2[i:i + BATCH_SIZE]
        batch_y = y[i:i + BATCH_SIZE]

        loss = trainer.train_step(batch_x1, batch_x2, batch_y)
        avg_loss += loss

    avg_loss /= len(x1) / BATCH_SIZE
    print("Epoch {}: average loss = {}".format(epoch + 1, avg_loss))
    show.append(avg_loss)

plt.plot(show)
plt.show()
# 테스트

x1_test = torch.randn((BATCH_SIZE, 64, EMBEDDING_DIM))
x2_test = torch.randn((BATCH_SIZE, 64, EMBEDDING_DIM))
dist = trainer.test_step(x1_test, x2_test)

print("Test results: distance = {}".format(dist))
