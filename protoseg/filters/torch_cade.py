import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing
from torch.utils import data

cade = None


def torch_cade(img, num_angles=8, distance=5, epochs=2, background_min=0, background_max=100, learn_rate=1, reinit=False):
    global cade
    if cade is None:
        cade = CADE(img, num_angles=num_angles, distance=distance,
                    epochs=epochs, learn_rate=learn_rate, reinit=reinit)
    return cade(img)


class Model(nn.Module):
    def __init__(self, cols, rows):
        super(Model, self).__init__()
        self.cols = cols
        self.rows = rows
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(4),
            nn.ReLU())
        self.layer3 = nn.Conv1d(4, 2, kernel_size=1, stride=1, padding=0)
        self.layer5 = nn.Conv1d(4, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        out = self.layer5(out)
        out = out.view(-1)
        return out

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
            torch.nn.init.uniform_(m.weight.data)
            torch.nn.init.uniform_(m.bias.data)

class Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CADE():
    torch.multiprocessing.set_start_method('forkserver', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, img, num_angles=8, distance=5, epochs=2, background_min=0, background_max=1, learn_rate=1, reinit=False):
        self.img = img
        if len(img.shape) == 3:
            self.img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        self.rows = img.shape[0]
        self.cols = img.shape[1]
        self.num_angles = num_angles
        self.distance = distance
        self.epochs = epochs
        self.background_min = background_min
        self.background_max = background_max
        self.reinit = reinit
        self.angles = np.linspace(
            0, (360.0 - 360.0 / self.num_angles), self.num_angles)
        self.model = Model(self.cols, self.rows)
        self.model = self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learn_rate)

    def rotationFilter(self, angle, distance):
        rows = 2*distance+1
        cols = 2*distance+1
        filt = np.zeros((rows, cols))
        filt[2*distance] = 0.5
        filt[distance] = -0.5
        M = cv2.getRotationMatrix2D((cols, rows), angle, 1)
        return cv2.warpAffine(filt, M, (cols, rows))

    def neighbourhood(self, img, angle, distance):
        return cv2.filter2D(img, -1, kernel=self.rotationFilter(angle, distance))

    def dataset(self, x, y):
        for i in range(len(x)):
            yield (x[i], y[i])

    def cade(self, img, neighbours):
        preds = []
        for neighbour in neighbours:
            if self.reinit is True:
                self.model.apply(self.model.weights_init)
            distr = list(map(list, zip(img.ravel(), neighbour.ravel())))
            distr = np.atleast_3d(distr)
            x = np.array(distr)
            y = np.zeros(len(x))
            x_fake = np.random.uniform(
                self.background_min, self.background_max, x.shape)
            y_fake = np.ones(y.shape)
            x_full = np.concatenate((x, x_fake), axis=0)
            y_full = np.concatenate((y, y_fake), axis=0)
       
            dataloader = data.DataLoader(
                Dataset(x_full, y_full), batch_size=25000, shuffle=True
            )
            for _ in range(self.epochs):
                for (x_,y_) in dataloader:
                    self.train_img(x_, y_)

            self.model.eval()
            x = torch.from_numpy(x).float().to(self.device)
            prob = self.model(x)
            prob = prob.cpu().detach().numpy()
            prob = prob.reshape(img.shape)
            preds.append(prob)
        return preds

    def train_img(self, x, y):
        x = x.float().to(self.device)
        y = y.float().to(self.device)
        # Forward pass
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def batchify(self, imgs):
        img_batch = np.array(imgs)
        img_batch = np.expand_dims(img_batch, axis=3)
        img_batch = np.transpose(img_batch, axes=[0, 3, 1, 2])
        return img_batch

    def __call__(self, img=None):
        if img is None:
            img = self.img
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.reinit is True:
            self.model.apply(self.model.weights_init)
        neighbours = []
        for angle in self.angles:
            neighbourhood_ = self.neighbourhood(img, angle, self.distance)
            neighbours.append(neighbourhood_)

        preds = self.cade(img, neighbours)

        pred = np.sum(np.array(preds), axis=0)
        pred = pred / (0.0001+pred.max())
        pred = pred * 255
        return pred.astype(np.uint8)
