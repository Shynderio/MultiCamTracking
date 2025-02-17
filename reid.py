from torchreid.data.transforms import build_transforms
from PIL import Image
import torchreid
import torch
from torchreid import metrics


class REID():
    def __init__(self, model_path):
        self.use_gpu = torch.cuda.is_available()
        print("use_gpu: ", self.use_gpu)
        if (model_path in ['resnet50', 'resnet50mid', 'hacnn', 'pcb_p6', 'pcb_p4', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0', 'osnet_ain_x1_0']):
            self.model = torchreid.models.build_model(
                name=model_path,
                num_classes=1,  # human
                loss='softmax',
                pretrained=True,
                use_gpu=self.use_gpu
            )
        else:
            self.model = torchreid.models.build_model(
                name='resnet50',
                num_classes=1,  # human
                loss='softmax',
                pretrained=True,
                use_gpu=self.use_gpu
            )
            torchreid.utils.load_pretrained_weights(self.model, model_path)
        if self.use_gpu:
            self.model = self.model.cuda()
        _, self.transform_te = build_transforms(
            height=256, width=128,
            random_erase=False,
            color_jitter=False,
            color_aug=False
        )
        self.dist_metric = 'euclidean'
        self.model.eval()

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _features(self, imgs):
        f = []
        for img in imgs:
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            img = self.transform_te(img)
            img = torch.unsqueeze(img, 0)
            if self.use_gpu:
                img = img.cuda()
            features = self._extract_features(img)
            features = features.data.cpu()  # tensor shape=1x2048
            f.append(features)
        f = torch.cat(f, 0)
        return f

    def _append_features(self, allfeatures, img):
        
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = self.transform_te(img)
        img = torch.unsqueeze(img, 0)
        if self.use_gpu:
            img = img.cuda()
        features = self._extract_features(img)
        features = features.data.cpu()

        # if (allfeatures is None):
        #     allfeatures = torch.cat(([features]), 0)
        # allfeatures = torch.cat((allfeatures, features), 0)
        if allfeatures is None:
            allfeatures = torch.cat(([features]), 0)
        else:
            allfeatures = torch.cat((allfeatures, features), 0)

        if allfeatures.size(0) > 20:
            allfeatures = allfeatures[-20:]
        return allfeatures
    
    def _cat_features(self, feats1, feats2):
        return torch.cat((feats1, feats2), 0)



    def compute_distance(self, qf, gf):
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        # print(distmat.shape)
        return distmat.numpy()


if __name__ == '__main__':
    reid = REID()