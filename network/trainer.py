import torch
from .discriminator import Adaptor, Classifier
from .feature_transformer import get_TRR
from tqdm.auto import tqdm
import os
import numpy as np
from sklearn.metrics import average_precision_score
from torch.utils.tensorboard import SummaryWriter

class Trainer(torch.nn.Module):
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.w_adaptor = opt.w_adaptor
        self.in_channels = opt.in_channels
        if self.w_adaptor:
            self.adaptor = Adaptor(opt.in_channels, opt.out_channels, opt.n_layers)
            self.optimizer_adaptor = torch.optim.Adam(self.adaptor.parameters(), lr=opt.lr)
            self.scheduler_adaptor = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adaptor, T_max=10, eta_min=0,
                                                                           last_epoch=-1, verbose=False)

        self.classifier = Classifier(opt.out_channels)
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters(), lr=opt.lr)
        self.scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_classifier, T_max=10,
                                                                             eta_min=0, last_epoch=-1, verbose=False)

        self.loss_func = torch.nn.BCEWithLogitsLoss()

        self.num_epochs = opt.num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.batchsize = opt.batchsize
        self.std = opt.std
        self.outdir = "{}/{}".format(opt.ckpt_dir, opt.train_name)
        self.writer = SummaryWriter(f"logs/{opt.train_name}")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.iter = 0

    def feature_perturbator(self, feature):
        noise = torch.cat([
            torch.normal(mean=0, std=self.std, size=(1, self.in_channels))
            for _ in range(self.batchsize)], dim=0).to(self.device)
        return noise + feature

    def train(self, train_dataloader, test_dataloader):
        best_record = None
        for epoch in range(self.num_epochs):
            self._train_discriminator(train_dataloader)
            ap = self.vanilla_eval(test_dataloader)

            if best_record is None:
                best_record = ap
                self.save()
            else:
                if ap > best_record:
                    best_record = ap
                    self.save()
            print(f"----- {epoch} ap:{round(ap, 4)}(MAX:{round(best_record, 4)})")
        self.writer.close()
        return best_record

    def _train_discriminator(self, train_dataloader):

        if self.w_adaptor:
            self.adaptor.to(self.device)
            self.adaptor.train()
        self.classifier.to(self.device)
        self.classifier.train()

        for images, labels in tqdm(train_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            synthetic_labels = torch.ones(size=labels.shape).to(self.device)
            all_labels = torch.cat([labels, synthetic_labels])

            TRRs = get_TRR(images, self.device).detach()
            synthetic_TRRs = self.feature_perturbator(TRRs)
            all_TRRs = torch.cat([TRRs, synthetic_TRRs])
            preds = self.classifier(self.adaptor(all_TRRs)).reshape(all_labels.shape)

            loss = self.loss_func(preds, all_labels)
            if self.w_adaptor:
                self.optimizer_adaptor.zero_grad()
            self.optimizer_classifier.zero_grad()

            loss.backward()

            if self.w_adaptor:
                self.optimizer_adaptor.step()
            self.optimizer_classifier.step()
            self.scheduler_adaptor.step()
            self.optimizer_classifier.step()
            self.writer.add_scalar("loss", loss.detach().cpu(), self.iter)
            self.iter += 1

    def vanilla_eval(self, test_dataloader):
        if self.w_adaptor:
            self.adaptor.eval()
        self.classifier.eval()

        total_image_pred = np.array([])
        total_image_gt = np.array([])
        tbar = tqdm(test_dataloader)
        with torch.no_grad():
            for images, labels in tbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                TRR = get_TRR(images, self.device).detach()
                pred = self.classifier(self.adaptor(TRR)).sigmoid()

                total_image_pred = np.append(total_image_pred, pred.detach().cpu().numpy())
                total_image_gt = np.append(total_image_gt, labels.detach().cpu().numpy())

            ap = average_precision_score(total_image_gt, total_image_pred)
        return ap

    def save(self):
        if self.w_adaptor:
            torch.save(
                {
                    'adaptor': self.adaptor.state_dict(),
                    'discriminator': self.classifier.state_dict()
                }, r"{}/all_model_best.pth".format(self.outdir)
            )
        else:
            torch.save(
                {
                    'discriminator': self.classifier.state_dict()
                }, r"{}/all_model_best.pth".format(self.outdir)
            )
