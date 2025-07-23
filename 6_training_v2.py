import os, shutil, json, cv2, pandas as pd, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import datetime
import glob
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append('../PyTorch_Torchvision')
from PyTorch_Torchvision.utils import collate_fn
from PyTorch_Torchvision.engine import train_one_epoch
from XBG_scripts.utils.evaluation import evaluate_xbg

class CustomDataset(Dataset):  # https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

    def __init__(self, img_dir, label_dir, mean, std):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_labels = pd.DataFrame(sorted(os.listdir(self.label_dir)))
        print(f'Dataset initialized with {len(self.img_labels)} images')

        if mean != None and std != None:
            print(f'Dataset normalized using mean and std\n')
            self.std = torch.tensor(mean)
            self.mean = torch.tensor(std)
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=self.mean,
                                                                      std=self.std)])
        else:
            print(f'Dataset not normalized\n')
            self.transform = transforms.ToTensor()


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_path = os.path.splitext(img_path)[0] + '.jpg'
        label_path = os.path.join(self.label_dir, self.img_labels.iloc[idx, 0])

        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path)
        # plt.imshow(img)
        # plt.show()
        img = self.transform(img)
        # plt.imshow(img.permute(1, 2, 0))
        # plt.show()

        with open(label_path) as f:
            data = json.load(f)
            bboxes = data['bboxes']
            keypoints = data['keypoints']

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)

        file_name = self.img_labels.iloc[idx, 0]

        return img, target, file_name

    def __len__(self):
        return len(self.img_labels)



def get_model(log_db, num_keypoints):
    anchor_generator = AnchorGenerator(sizes=(15, 75, 125, 250, 500),
                                       aspect_ratios=(0.4, 0.6, 0.8, 1.1, 1.6))

    log_db = log(log_db, "anchor_generator sizes", anchor_generator.sizes)
    log_db = log(log_db, "anchor_generator aspect ratios", anchor_generator.aspect_ratios[0])

    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None,
                                                                   weights_backbone='ResNet50_Weights.DEFAULT',
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes=2,
                                                                   rpn_anchor_generator=anchor_generator,
                                                                   trainable_backbone_layers=5)

    return model, anchor_generator, log_db

def log(log_db, parameter, value):
    log_db_temp = pd.DataFrame({"Properties": [parameter], "Value": [value]})
    log_db = pd.concat([log_db, log_db_temp])
    return log_db

def log_train(db_train, epoch, lr, loss_value, loss_keypoint, dist_mean, dist_area_mean, score_mean, distNaN_mean, eval_num):
    log_db_temp = pd.DataFrame({"epoch": [epoch],
                                "lr": [lr],
                                "loss": [loss_value],
                                "loss_keypoint": [loss_keypoint],
                                "distance": [dist_mean],
                                "distance/area": [dist_area_mean],
                                "distanceNan": [distNaN_mean],
                                "score": [score_mean],
                                "eval_num": [eval_num]})

    db_train = pd.concat([db_train, log_db_temp])
    return db_train

def calculate_mean_std(dir_path, img_normalization):
    if img_normalization == True:
        mean_sum = []
        std_sum = []
        for i, image in enumerate(glob.glob(os.path.join(dir_path,'*.jpg'))):
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            img_tensor = transforms.ToTensor()(img)
            mean, std = img_tensor.mean((1, 2)), img_tensor.std((1, 2))
            mean_sum.append(mean.numpy())
            std_sum.append(std.numpy())
        mean_array = np.array(mean_sum)
        std_array = np.array(std_sum)
        mean = [mean_array[:, 0].mean(), mean_array[:, 1].mean(), mean_array[:, 2].mean()]
        std = [std_array[:, 0].mean(), std_array[:, 1].mean(), std_array[:, 2].mean()]
    else:
        mean = None
        std = None
    return mean, std

log_db = pd.DataFrame(columns=['Properties','Value'])
db_train = pd.DataFrame(columns=["epoch",
                                 "lr",
                                 "loss",
                                 "loss_keypoint",
                                 "distance",
                                 "distance/area",
                                 "distanceNan",
                                 "score",
                                 "eval_num"])

train_images_dir = '/mnt/GSS_XBG/GCP_Publication/Datasets/GCP_Nossen/train/images/augmented/images'
train_labels_dir = '/mnt/GSS_XBG/GCP_Publication/Datasets/GCP_Nossen/train/images/augmented/labels'
eval_images_dir = '/mnt/GSS_XBG/GCP_Publication/Datasets/GCP_Nossen/validation/images'
eval_labels_dir = '/mnt/GSS_XBG/GCP_Publication/Datasets/GCP_Nossen/validation/labels'
log_db = log(log_db, "train_images_dir", train_images_dir)
log_db = log(log_db, "train_labels_dir", train_labels_dir)
log_db = log(log_db, "eval_images_dir", eval_images_dir)
log_db = log(log_db, "eval_labels_dir", eval_labels_dir)


''' Parameters '''
num_epochs = 1000
batch_size_train = 53
batch_size_eval = 1
num_keypoints = 1

lr=0.0015

momentum=0.9
weight_decay=0

step_size=100
gamma=0.85

img_normalization = False
''''''

print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')

log_db = log(log_db, "num_keypoints", num_keypoints)
log_db = log(log_db, "lr", lr)
log_db = log(log_db, "momentum", momentum)
log_db = log(log_db, "weight_decay", weight_decay)
log_db = log(log_db, "step_size", step_size)
log_db = log(log_db, "gamma", gamma)

mean_train, std_train = calculate_mean_std(train_images_dir, img_normalization)
mean_eval, std_eval = calculate_mean_std(eval_images_dir, img_normalization)

log_db = log(log_db, "img_normalization", img_normalization)
log_db = log(log_db, "mean_train", mean_train)
log_db = log(log_db, "std_train", std_train)
log_db = log(log_db, "mean_eval", mean_eval)
log_db = log(log_db, "std_eval", std_eval)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
log_db = log(log_db, "Device", device)
print(f'\nDevice: {device}')
print(f'GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')

print(f'\nLoading training dataset')
dataset_train = CustomDataset(train_images_dir, train_labels_dir, mean_train, std_train)
data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
log_db = log(log_db, "train batch", batch_size_train)

print(f'Loading validation dataset')
dataset_eval = CustomDataset(eval_images_dir, eval_labels_dir, mean_eval, std_eval)
data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size_eval, shuffle=False, collate_fn=collate_fn)
log_db = log(log_db, "eval batch", batch_size_eval)

print(f'Loading model')
model, anchor_generator, log_db = get_model(log_db, num_keypoints)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

log_db = log(log_db, "num_epochs", num_epochs)
log_db = log(log_db, "optimizer", str(optimizer).split()[0])

path_parent = datetime.datetime.now().strftime("%Y%m%d_%H%M") + "_Keypoint"
if os.path.exists(path_parent):
    shutil.rmtree(path_parent, ignore_errors=False)
os.mkdir(path_parent)

log_db.to_csv(path_parent + '/parameters.txt', sep=" ", index=False)

for epoch in range(num_epochs):
    metrics_train = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
    lr_scheduler.step()
    loss_value = round(metrics_train.meters['loss'].global_avg, 5)
    loss_keypoint = round(metrics_train.meters['loss_keypoint'].global_avg, 5)
    dist_mean, score_mean, dist_area_mean, distNaN_mean, eval_num = evaluate_xbg(model, data_loader_eval, device)
    lr = round(metrics_train.meters['lr'].avg, 5)

    db_train = log_train(db_train, epoch, lr, loss_value, loss_keypoint, dist_mean, dist_area_mean, score_mean, distNaN_mean, eval_num)

    if eval_num > 80 and epoch > 10 and score_mean > 0.95:
        if dist_mean <= np.nanmin(db_train['distance'].values):
            for filename in glob.glob(path_parent + "/*__DstPx-*"):
                os.remove(filename)
            torch.save({
            'epoch': epoch, 'model': model, 'model_state_dict': model.state_dict(), 'loss': loss_value,
            'anchor_generator': anchor_generator}, path_parent + '/' + str(datetime.date.today()) + '_e' + str(epoch) + '_n' + str(eval_num) + '__DstPx-' + str(round(dist_mean, 3)) + '_DArea-' + str(round(dist_area_mean, 3)) + '_Score-' + str(round(score_mean, 3)) + '_rcnn-resnet50.pth')

        if score_mean >= db_train['score'].max():
            for filename in glob.glob(path_parent + "/*__Score-*"):
                os.remove(filename)
            torch.save({
            'epoch': epoch, 'model': model, 'model_state_dict': model.state_dict(), 'loss': loss_value,
            'anchor_generator': anchor_generator}, path_parent + '/' + str(datetime.date.today()) + '_e' + str(epoch) + '_n' + str(eval_num) + '__Score-' + str(round(score_mean, 3)) + '_DstPx-' + str(round(dist_mean, 3)) + '_DArea-' + str(round(dist_area_mean, 3)) + '_rcnn-resnet50.pth')

        if dist_area_mean <= np.nanmin(db_train['distance/area'].values):
            for filename in glob.glob(path_parent + "/*__DArea-*"):
                os.remove(filename)
            torch.save({
            'epoch': epoch, 'model': model, 'model_state_dict': model.state_dict(), 'loss': loss_value,
            'anchor_generator': anchor_generator}, path_parent + '/' + str(datetime.date.today()) + '_e' + str(epoch) + '_n' + str(eval_num) + '__DArea-' + str(round(dist_area_mean, 3)) + '_DstPx-' + str(round(dist_mean, 3)) + '_Score-' + str(round(score_mean, 3)) + '_rcnn-resnet50.pth')

    fig, ax = plt.subplots()
    ax.plot(db_train['loss_keypoint'].values, color="orange", label="Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Training loss")
    ax2 = ax.twinx()
    ax2.plot(db_train['distance'].values, label='Distance')
    ax2.set_ylabel("Distance [Pixels]")
    if epoch>31:
        ax2.set_ylim(0, np.nanmax(db_train['distance'].values[-30:])*2.5)
    ax.grid()
    fig.savefig(path_parent + "/accuracy.png")
    fig.clear()
    plt.close(fig)

    db_train.to_csv(path_parent + '/log_training.txt', sep= " ", index = False)

print("Training process finalized")