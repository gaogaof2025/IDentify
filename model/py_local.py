import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard.summary import pr_curve_raw

from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .modules import *
import torch.nn.functional as F
from torch.nn import init

_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.simatt = SimAM()
        self.relu = nn.LeakyReLU(.02)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = self.simatt(x)
        x = self.relu(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.relu(x)

        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


# from NFC import NFC
from NFC_Enhance import nfc


def l2_normalize(x, dim=-1):
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + 1e-8)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'

class build_transformer(nn.Module):
    def __init__(self, num_classes, num_clothes, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.image_size = cfg.INPUT.SIZE_TRAIN
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.sie_coe = cfg.MODEL.SIE_COE
        self.num_clothes = num_clothes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.classifier_cloth = nn.Linear(self.in_planes, self.num_clothes, bias=False)
        self.classifier_cloth.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        #-------------------------------------
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)

        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)

        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)

        self.classifier_3 = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)


        self.gap_f = GeM()

        self.former_LRU_2 = LocalRefinementUnits(dim=self.in_planes, out_dim=self.in_planes)
        self.former_LRU_3 = LocalRefinementUnits(dim=self.in_planes_proj, out_dim=self.in_planes_proj)

        # --------------------------------------------------------------


        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

        self.PRA = FAA_ViT(dim=self.in_planes_proj)
        self.PRA_2 = FAA_ViT(dim=self.in_planes)
        # self.eda = EfficientAdditiveAttention(in_dims=768, token_dim=768)

        self.cam = CAM_Module(self.in_planes_proj)
        self.cam_1 = CAM_Module(self.in_planes)
        #
        self.sam = SpatialAttention()


        # self.simatt = SimAM()
        #
        # self.gf = GlobalFilter(self.in_planes, h=17, w=9)
        # self.gf_proj = GlobalFilter(self.in_planes_proj, h=17, w=9)

    def forward(self, x=None, label=None, get_image=False, get_text=False, noise_img=None):
        if get_text == True:
            prompts = self.prompt_learner(label)
            text_features = (self.text_encoder(prompts, self.prompt_learner.tokenized_prompts))
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            # image_features = self.gf(image_features)
            # image_features_proj = self.gf_proj(image_features_proj)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)

            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            cv_embed = None

            # (32, 289, 768)           # (32, 289, 512)
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            # image_features = NFC(image_features,5,5)
            # image_features_proj = NFC(image_features_proj,5,5)
            image_features = nfc(image_features, "knn", 'mean', 4)  # B L D
            image_features_proj = nfc(image_features_proj, "knn", 'mean', 4)
            if noise_img is not None:
                noise_image_features_last, noise_image_features, noise_image_features_proj = self.image_encoder(
                    noise_img, cv_embed)
                # (32, 289, 768)        # (32, 289, 512)
                # -------------------------------------------------------------------------------------
                # noise_image_features = nfc(noise_image_features, "knn", 'mean', 7)  # B L D
                # noise_image_features_proj = nfc(noise_image_features_proj, "knn", 'mean', 7)
                #######################################################
                # 计算注意力权重
                attention_weights = torch.sigmoid(
                    torch.sum(image_features * noise_image_features, dim=-1, keepdim=True))

                # 动态调整特征
                feature_diff = image_features - noise_image_features
                feature_proj_diff = image_features_proj - noise_image_features_proj
                cloth_noise = attention_weights * feature_diff  # (32, 289, 768)
                cloth_proj_noise = attention_weights * feature_proj_diff

                image_features = image_features - cloth_noise  # (32, 289, 768)
                image_features_proj = image_features_proj - cloth_proj_noise  # (32, 289, 512)
                ####################################################

            # image_features_proj = self.PRA(image_features_proj)
            # image_features_last = self.PRA_2(image_features_last)
            # image_features = self.PRA_2(image_features)
            # -------------------------------------------------------

            image_features_proj = self.cam(image_features_proj)
            image_features_proj = self.sam(image_features_proj)
            image_features = self.cam_1(image_features)
            image_features = self.sam(image_features)

            img_feature_last = image_features_last[:, 0]
            img_feature = (image_features[:, 0])
            img_feature_proj = (image_features_proj[:, 0])


        #---------------------------------------------------
        mid_f = image_features[:, 1:, :]
        mid_proj_f = image_features_proj[:, 1:, :]

        mid_f = mid_f.permute(0, 2, 1).contiguous().view(image_features_last.shape[0], 768, 24, 12)
        mid_f = self.former_LRU_2(mid_f)
        local_former_2 = self.gap_f(mid_f)
        global_feat_2 = local_former_2.view(image_features_last.shape[0], -1)
        feat_2 = self.bottleneck_2(global_feat_2)

        mid_proj_f = mid_proj_f.permute(0, 2, 1).contiguous().view(image_features_last.shape[0], 512, 24, 12)
        mid_proj_f = self.former_LRU_3(mid_proj_f)
        local_former_3 = self.gap_f(mid_proj_f)
        global_feat_3 = local_former_3.view(image_features_last.shape[0], -1)
        feat_3 = self.bottleneck_3(global_feat_3)

        #---------------------------------------------------


        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_cloth_score = self.classifier_cloth(feat)
            cls_score_proj = self.classifier_proj(feat_proj)

            cls_score_2 = self.classifier_2(feat_2)
            cls_score_3 = self.classifier_3(feat_3)

            return [cls_score, cls_score_proj], [img_feature_last, img_feature,
                                                 img_feature_proj], img_feature_proj, cls_cloth_score, feat_2, cls_score_2, feat_3, cls_score_3

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class LocalRefinementUnits(nn.Module):
    def __init__(self, dim, out_dim=768, kernel=1, choice=True):
        super().__init__()
        self.LRU = choice
        self.channels = dim
        self.out_dim = out_dim
        self.dwconv = nn.Conv2d(self.channels, self.channels, kernel, 1, padding=0, groups=self.channels)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.ptconv = nn.Conv2d(self.channels, self.out_dim, 1, 1)
        self.bn2 = nn.BatchNorm2d(self.out_dim)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act = nn.ReLU()

    def forward(self, x):
        if self.LRU:
            x = self.act1(self.bn1(self.dwconv(x)))
            x = self.act2(self.bn2(self.ptconv(x)))
        else:
            x = self.act2(self.bn2(self.ptconv(x)))
        return x


def make_model(cfg, num_class, num_clothes):
    model = build_transformer(num_class, num_clothes, cfg)
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        # ctx_init = "A photo of a X X X X X X X X X person."
        ctx_init = "A photo of a X X X X person with X X cloth."
        # ctx_init = "A photo of a person with X X X X X clothing and X X features."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        # print(f"prefix shape: {prefix.shape}") #  torch.Size([64, 5, 512])
        # print(f"cls_ctx shape: {cls_ctx.shape}") #  torch.Size([64, 4, 512])
        # print(f"suffix shape: {suffix.shape}") #  torch.Size([64, 68, 512])

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

