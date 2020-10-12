# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from visdom import Visdom
import time
import cv2
from torchvision import transforms, models


class DataSet(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.img_info_dir = 'VOC2012/ImageSets/Segmentation/'
        self.img_dir = 'VOC2012/JPEGImages/'
        self.seg_cls_dir = 'VOC2012/SegmentationClass/'
        self.seg_obj_dir = 'VOC2012/SegmentationObject/'
        self.re_size = 388
        self.border = (572 - 388) // 2
        self.seg_cls = True
        if args.assignment == 'seg_cls':
            self.feature_num = 21
        elif args.assignment == 'seg_obj':
            self.feature_num = 40
            self.seg_cls = False
        else:
            print("should input right assignment, it is one in ['seg_cls', 'seg_obj']")
            exit()
        train_set, test_set = self.train_test()
        self.data_set = train_set if args.work in ['train', 'finetune'] else test_set  # 选定数据集
        # cls = []
        # for obj in train_set:
        #     seg_cls = Image.open(obj['seg_obj'])
        #     img_data = np.asarray(seg_cls) + 1
        #     for i in img_data:
        #         for j in i:
        #             if j not in cls:
        #                 cls.append(j)
        # print(cls)
        # for obj in self.data_set[100:]:
        #     ori_img = Image.open(obj['img_path'])
        #     seg_cls = Image.open(obj['seg_cls'])
        #     seg_obj = Image.open(obj['seg_obj'])
        #     plt.subplot(131)
        #     plt.imshow(ori_img)
        #     plt.subplot(132)
        #     plt.imshow(seg_cls)
        #     plt.subplot(133)
        #     plt.imshow(seg_obj)
        #     plt.show()

    def train_test(self):
        label = ['train.txt', 'val.txt']
        result = []
        for _i in label:
            sub_result = []
            label_data = open(self.img_info_dir + _i)
            label_lines = label_data.readlines()
            label_data.close()
            for _j in label_lines:
                sub_result.append({'img_path': self.img_dir + _j[:-1] + '.jpg',
                                   'seg_cls': self.seg_cls_dir + _j[:-1] + '.png',
                                   'seg_obj': self.seg_obj_dir + _j[:-1] + '.png'})
            result.append(sub_result)
        return result

    def __getitem__(self, item):
        obj = self.data_set[item]
        ori_img = np.array(Image.open(obj['img_path']).resize((self.re_size, self.re_size)))
        # ori_img = cv2.imread(obj['img_path'], cv2.IMREAD_COLOR)
        # ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        # ori_img = cv2.resize(ori_img, (self.re_size, self.re_size))
        border_img = cv2.copyMakeBorder(ori_img, self.border, self.border, self.border, self.border, cv2.BORDER_REFLECT)
        # print('border_img', border_img.size)
        if self.transform is not None:
            border_img = self.transform(border_img)
        if self.seg_cls:
            label = np.array(Image.open(obj['seg_cls']).resize((self.re_size, self.re_size))) + 1
        else:
            label = np.array(Image.open(obj['seg_obj']).resize((self.re_size, self.re_size))) + 1
        return border_img, label

    def __len__(self):
        return len(self.data_set)


def conv_actv_pool(in_chls, out_chls, pool=False, upsample=False):
    layers = []
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2))
    layers.append(nn.Conv2d(in_chls, out_chls, kernel_size=3))
    layers.append(nn.ReLU(inplace=True))
    if upsample:
        layers.append(nn.ConvTranspose2d(out_chls, out_chls // 2, kernel_size=2, stride=2))
    return layers


def basic_block(in_chls, out_chls, downsample=False, upsample=False):
    blocks = []
    if downsample and upsample:
        blocks.extend(conv_actv_pool(in_chls, out_chls, pool=True))
        blocks.extend(conv_actv_pool(out_chls, out_chls, upsample=True))
    elif downsample:
        blocks.extend(conv_actv_pool(in_chls, out_chls, pool=True))
        blocks.extend(conv_actv_pool(out_chls, out_chls))
    elif upsample:
        blocks.extend(conv_actv_pool(in_chls, out_chls))
        blocks.extend(conv_actv_pool(out_chls, out_chls, upsample=True))
    else:
        blocks.extend(conv_actv_pool(in_chls, out_chls))
        blocks.extend(conv_actv_pool(out_chls, out_chls))
    return blocks


class UNet(nn.Module):
    def __init__(self, output_feature_num):
        super(UNet, self).__init__()
        self.left_1 = nn.Sequential(*basic_block(3, 64))
        self.left_2 = nn.Sequential(*basic_block(64, 128, downsample=True))
        self.left_3 = nn.Sequential(*basic_block(128, 256, downsample=True))
        self.left_4 = nn.Sequential(*basic_block(256, 512, downsample=True))
        self.bottom = nn.Sequential(*basic_block(512, 1024, downsample=True, upsample=True))
        self.right_4 = nn.Sequential(*basic_block(1024, 512, upsample=True))
        self.right_3 = nn.Sequential(*basic_block(512, 256, upsample=True))
        self.right_2 = nn.Sequential(*basic_block(256, 128, upsample=True))
        self.right_1 = nn.Sequential(*basic_block(128, 64))
        self.output = nn.Conv2d(64, output_feature_num, kernel_size=1, stride=1)

    def forward(self, x):
        output_left_1 = self.left_1(x)
        output_left_2 = self.left_2(output_left_1)
        output_left_3 = self.left_3(output_left_2)
        output_left_4 = self.left_4(output_left_3)
        output_bottom = self.bottom(output_left_4)
        output_right_4 = self.right_4(torch.cat([output_left_4[:, :, 4:60, 4:60], output_bottom], dim=1))
        output_right_3 = self.right_3(torch.cat([output_left_3[:, :, 16:120, 16:120], output_right_4], dim=1))
        output_right_2 = self.right_2(torch.cat([output_left_2[:, :, 40:240, 40:240], output_right_3], dim=1))
        output_right_1 = self.right_1(torch.cat([output_left_1[:, :, 88:480, 88:480], output_right_2], dim=1))
        output = self.output(output_right_1)
        return output


class MultiWorks:
    def __init__(self, load_model_path=None):
        self.start_time = time.time()  # 开始时间，用于输出时长
        self.load_model_path = load_model_path  # 微调、测试和预测时提供模型加载路径
        # 数据集
        # self.data_transform = transforms.Compose([transforms.ToTensor(),
        #                                           transforms.Normalize(tuple(self.data_mean), tuple(self.data_std))])
        self.data_transform = transforms.Compose([transforms.ToTensor()])
        self.data_set = DataSet(transform=self.data_transform)

        # 执行选择任务，train; test; finetune; predict;
        if args.work not in ['train', 'test', 'finetune', 'predict']:
            print("The args.work should be one of ['train', 'test', 'finetune', 'predict']")
        elif args.work == "train":  # 训练
            self.train()
        elif self.load_model_path is None:
            print("Please input 'load_model_path'")
        elif args.work == "test":  # 测试
            self.test()
        elif args.work == "finetune":  # 调模型
            self.finetune()
        elif args.work == "predict":  # 预测
            self.predict()

    def train(self):
        data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=args.batch_size, shuffle=True)
        # 输出开始及数据集大小
        print(f"Start Train!  len_dataset: {self.data_set.__len__()}")

        model = UNet(self.data_set.feature_num).to(device).train()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        current_lr = args.lr  # 步长
        # optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, betas=(0.9, 0.999))
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化器
        #
        collect_loss = [['epoch', 'current_lr', 'epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss']]
        epoch_count = []
        loss_record = []
        cost_time_record = []
        for i in range(args.epochs):  # 开始训练
            epoch_loss = []  # 每个epoch的loss
            for index, (img, label) in enumerate(data_loader):
                # self.interval_plot(img, label)
                img = img.to(device)  # 图片输入设备
                label = torch.from_numpy(label.numpy().astype('int64')).to(device)  # 标签输入设备
                # label = label.to(device)
                optimizer.zero_grad()  # 优化器梯度清零
                output = model(img)  # 计算模型输出
                loss = criterion(output, label)  # 计算loss值
                epoch_loss.append(loss.item())  # 采集epoch内的loss
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 优化器更新模型参数
            # if i % 20 == 0:
            #     self.show_face_landmarks(img, i, labels=label)
            epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
            epoch_max_loss = max(epoch_loss)
            epoch_min_loss = min(epoch_loss)
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss])
            cost_time_record.append(time.time() - self.start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart2', opts=opts2)
            # 采集epoch序数和每个patch的平均、最大、最小loss
            collect_loss.append([i, current_lr, epoch_mean_loss, epoch_max_loss, epoch_min_loss])  # 采集loss
            if i == 5:
                current_lr = args.lr
                optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化函数
            # if i == 15:
            #     current_lr = args.lr * 2000
            #     optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化函数

        # 保存模型和loss
        if args.save_model:  # 是否保存模型
            if not os.path.exists(args.save_directory):  # 新建保存文件夹
                os.makedirs(args.save_directory)
            save_model_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_epoch_' + str(i)
                                           + ".pt")
            save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_loss.csv')
            torch.save(model.state_dict(), save_model_path)
            self.writelist2csv(collect_loss, save_loss_path)
            print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def test(self):
        do_data_transform = transforms.Compose([transforms.ToTensor()])
        data_set = DataSet(args.train_set_ratio, transform=do_data_transform)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.test_batch_size, shuffle=False)
        # 输出数据集大小
        print(f"Start Test!  len_dataset: {self.data_set.__len__()}")
        model = ResNet([1, 1, 1]).to(device)
        model.load_state_dict(torch.load(self.load_model_path))  # 模型参数加载
        model.eval()  # 关闭参数梯度

        criterion = nn.MSELoss()  # SmoothL1Loss()

        collect_loss = [['epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss']]
        epoch_loss = []  # 每个epoch的loss
        for index, (img, label) in enumerate(data_loader):
            img_plot, label_plt = img.numpy(), label.numpy()
            img = img.to(device)  # 图片输入设备
            label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
            output = model(img)  # 前向传播计算模型输出
            loss = criterion(output, label)  # 计算loss值
            epoch_loss.append(loss.item())
            # 查看模型输出及标签
            self.show_face_landmarks(img_plot, output.cpu().detach().numpy(), labels=label_plt, patch_plot_num=1)
        epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
        epoch_max_loss = max(epoch_loss)
        epoch_min_loss = min(epoch_loss)
        print(f'mean_epoch_loss: {epoch_mean_loss}  max_batch_loss: {epoch_max_loss}  min_batch_loss: {epoch_min_loss}')
        collect_loss.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss])  # 采集loss
        save_loss_path = self.load_model_path[:-3] + '_test_result.csv'
        self.writelist2csv(collect_loss, save_loss_path)
        print(f'--Save complete!\n--save_loss_path: {save_loss_path}\n')
        print('Test complete!')

    def finetune(self):
        data_loader = torch.utils.data.DataLoader(self.data_set, batch_size=args.batch_size, shuffle=True)
        # 输出数据集大小
        print(f"Start Finetune!  len_dataset: {self.data_set.__len__()}")
        model = NetStage2().to(device).train()
        model.load_state_dict(torch.load(self.load_model_path))  # 模型参数加载
        criterion = nn.MSELoss()
        current_lr = args.lr  # 步长
        # optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, betas=(0.9, 0.999))
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # 优化器

        collect_loss = [['epoch', 'current_lr', 'epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss']]
        epoch_count = []
        loss_record = []
        cost_time_record = []
        for i in range(args.epochs):  # 开始训练
            epoch_loss = []  # 每个epoch的loss和
            for index, (img, label) in enumerate(data_loader):
                img = img.to(device)  # 数据输入设备
                label = torch.from_numpy(label.numpy().astype('float32')).to(device)  # 标签输入设备
                optimizer.zero_grad()  # 优化器梯度清零
                output = model(img)  # 前向传播计算模型输出
                loss = criterion(output, label)  # 计算loss值
                epoch_loss.append(loss.item())  # 采集epoch内的loss
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 优化器更新模型参数
            if i % 20 == 0:
                self.show_face_landmarks(img, i, labels=label)
            epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
            epoch_max_loss = max(epoch_loss)
            epoch_min_loss = min(epoch_loss)
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss])
            cost_time_record.append(time.time() - self.start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart2', opts=opts2)
            # 采集epoch序数和每个patch的平均、最大、最小loss
            collect_loss.append([i, current_lr, epoch_mean_loss, epoch_max_loss, epoch_min_loss])  # 采集loss

        # 保存模型和loss
        save_model_path = self.load_model_path[:-3] + '_finetune_' + str(i) + ".pt"
        save_loss_path = self.load_model_path[:-3] + '_finetune_' + str(i) + "_loss.csv"
        torch.save(model.state_dict(), save_model_path)
        self.writelist2csv(collect_loss, save_loss_path)
        print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Finetune complete!')

    def predict(self):
        predict_result = []
        # 输出数据集大小
        print(f"Start Predict!")
        model = ResNet([2, 2, 2]).to(device)
        model.load_state_dict(torch.load(self.load_model_path))  # 模型参数加载
        model.eval()
        train_set_mean, train_set_std = 114.63897582368799, 66.18579280506815

        while True:
            img_ori, k = self.capture_predict_face()
            if k == 27:
                print('Quit Predict!')
                break
            img_norm = ((img_ori - train_set_mean) / (train_set_std + 0.0000001)).astype('float32')
            img_tensor = transforms.ToTensor()(img_norm)
            img_s = np.expand_dims(img_tensor, axis=0)
            img = torch.from_numpy(img_s).to(device)  # 数据输入设备
            output = model(img)  # 前向传播计算模型输出
            self.show_face_landmarks([[img_ori]], output.cpu().detach().numpy(), labels=None, patch_plot_num=1)  # 查看
            predict_result.append([output.data])
            print(f'-predict_result: {output.data}')
        save_loss_path = self.load_model_path[:-3] + '_predict_result.csv'
        self.writelist2csv(predict_result, save_loss_path)
        print('Predict complete!')

    @staticmethod
    def writelist2csv(list_data, csv_name):  # 列表写入.csv
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for one_slice in list_data:
                csv_writer.writerow(one_slice)

    @staticmethod
    def interval_plot(y_hat, img_content, img_style):
        plt.subplot(131)
        plt.imshow(y_hat)
        plt.subplot(132)
        plt.imshow(img_content)
        plt.subplot(133)
        plt.imshow(img_style)
        plt.show()

    def show_face_landmarks(self, imgs, epoch, labels=None, outputs=None, plots=1):  # 绘制处理后的图片
        random_plt = random.sample(range(len(imgs)), plots)  # plots每个patch选择绘制的图片数
        for i in random_plt:
            img = imgs[i].cpu().detach().numpy().transpose(1, 2, 0)
            img = img * np.array(self.data_std) + np.array(self.data_mean)
            img = np.array(np.rint(img * 255), dtype='uint8')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)

            landmarks_l = labels[i].cpu().detach().numpy().reshape(-1, 2).T  # 预测值
            plt.scatter(landmarks_l[0], landmarks_l[1], s=3, c='r')
            if outputs is not None:
                landmarks_o = outputs[i].cpu().detach().numpy().reshape(-1, 2).T
                plt.scatter(landmarks_o[0], landmarks_o[1], s=3, c='b')
            plt.title(f'Epoch: {epoch}  Red-Label Blue-Predict')
            plt.show()

    @staticmethod
    def capture_predict_face():
        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()
            img_ori = img.copy()
            h, w, _ = img.shape
            c_x, c_y, rect_edge = int(w / 2), int(h / 2), int(min(h, w) * 0.3)
            cv2.rectangle(img, (c_x - rect_edge, c_y - rect_edge), (c_x + rect_edge, c_y + rect_edge), (0, 255, 0), 4)
            cv2.putText(img, 'Please put your face in the green rectangle.',
                        (int(h * 0.0), int(w * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 4)
            cv2.putText(img, "'Enter' to go on predict, 'Esc' to quit.",
                        (int(h * 0.08), int(w * 0.12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)
            cv2.imshow("img", img)
            k = cv2.waitKey(1)
            if k == 27:  # Esc
                img = []
                break
            elif k == 13:  # Enter
                capture_img = img_ori[c_y - rect_edge: c_y + rect_edge, c_x - rect_edge: c_x + rect_edge, :]
                img = cv2.resize(capture_img, (224, 224), interpolation=cv2.INTER_AREA)
                break
        cv2.destroyAllWindows(), cap.release()
        return img, k


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--train-set-ratio', type=float, default=0.8, metavar='N',
                        help='train percentage of all data (default: 0.8)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.99, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='save_stage2_model',
                        help='learnt models are saving here')
    parser.add_argument('--re-generate-data', type=bool, default=False,
                        help='if need to re generate data')
    parser.add_argument('--re-cal-mean-std', type=bool, default=False,
                        help='if need to re calculate dateset mean and std')
    parser.add_argument('--assignment', type=str, default='seg_cls',
                        help='seg_cls / seg_obj')
    parser.add_argument('--work', type=str, default='train',  # train, test, finetune, predict
                        help='training, test, predicting or finetuning')
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="Unet class segmentation")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss of mean/max/min in epoch',
        "xlabel": 'epoch',
        "ylabel": 'loss',
        "width": 600,
        "height": 400,
        "legend": ['mean_loss', 'max_loss', 'min_loss']
    }
    opts2 = {
        "title": 'cost time with epoch',
        "xlabel": 'epoch',
        "ylabel": 'time in second',
        "width": 400,
        "height": 300,
        "legend": ['cost time']
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MultiWorks(load_model_path='save_stage2_model/202005280642_train_epoch_99.pt')
    # d = DataSet()
    # for i in range(1000):
    #     a, b = d.__getitem__(i)
    #     a = Image.fromarray(a)
    #
    #     print(a.mode, b.mode)
    #     plt.subplot(131)
    #     plt.imshow(a)
    #     plt.subplot(132)
    #     plt.imshow(b)
    #     plt.show()
