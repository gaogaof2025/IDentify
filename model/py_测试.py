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

        self.cam = CAM_Module(self.in_planes_proj)
        self.cam_1 = CAM_Module(self.in_planes)
        #
        self.sam = SpatialAttention()

        self.cabm = CBAMLayer(channel=768)
        self.cabm_2 = CBAMLayer(channel=512)
        self.sc = StripeLocal(stripe_dim=256)  # 256
        self.stripe_bn = nn.BatchNorm1d(2304)

        self.stripe_bn_2 = nn.BatchNorm1d(1536)

        self.faa = FAA(in_channels=768)

        # self.simatt = SimAM()
        #
        # self.gf = GlobalFilter(self.in_planes, h=17, w=9)
        # self.gf_proj = GlobalFilter(self.in_planes_proj, h=17, w=9)

        # self.dwconv = nn.Sequential(
        #     nn.Conv1d(self.in_planes_proj, self.in_planes_proj, kernel_size=3, stride=1, padding=1,
        #               groups=self.in_planes_proj),
        #     nn.BatchNorm1d(self.in_planes_proj),
        #     nn.GELU()
        # )
        # self.dwconv_2 = nn.Sequential(
        #     nn.Conv1d(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1, groups=self.in_planes),
        #     nn.BatchNorm1d(self.in_planes),
        #     nn.GELU()
        # )

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
            # image_features_proj = self.cam(image_features_proj)
            # image_features_proj = self.sam(image_features_proj)
            # image_features = self.cam_1(image_features)
            # image_features = self.sam(image_features)

            token = image_features[:, 0:1, :]
            stripe_features = image_features[:, 1:, :]
            stripe_features = stripe_features.permute(0,2,1).contiguous().view(image_features_last.shape[0], 768, 24, 12)   # (32,768,24,12)


            stripe_features = self.sc(stripe_features)  # (32,2304,1,1)
            stripe_features = stripe_features.view(stripe_features.size(0), -1)  # (32,2304)
            stripe_features = self.stripe_bn(stripe_features)  # (32,2304)
            # stripe_features = stripe_features.view(32, 768, -1).permute(0,2,1)     # (32, 288, 768)
            # stripe_features = torch.cat([token, stripe_features], dim=1)     # (32, 289, 768)
            #

            token = image_features_proj[:, 0:1, :]
            stripe_features_proj = image_features_proj[:, 1:, :]
            stripe_features_proj = stripe_features_proj.permute(0,2,1).contiguous().view(image_features_last.shape[0], 512, 24, 12)   # (32,512,24,12)

            stripe_features_proj = self.sc(stripe_features_proj)  # (32,1536,1,1)
            stripe_features_proj = stripe_features_proj.view(stripe_features_proj.size(0), -1)  # (32,1536)
            stripe_features_proj = self.stripe_bn_2(stripe_features_proj)  # (32,1536)


            # stripe_features_proj = self.cabm_2(stripe_features_proj)
            # stripe_features_proj = stripe_features_proj.view(32,512, -1).permute(0, 2, 1)
            # stripe_features_proj = torch.cat([token, stripe_features_proj], dim=1)
            #-------------------------------------------------------------------------------------

            # image_features_proj = self.PRA(image_features_proj)
            #image_features_last = self.PRA_2(image_features_last)
            # image_features = self.PRA_2(image_features)
            # -------------------------------------------------------
            # image_features_proj = self.sam(image_features_proj)
            # image_features_proj = self.cam(image_features_proj)
            # image_features = self.sam(image_features)
            # image_features = self.cam_1(image_features)

            img_feature_last = image_features_last[:, 0]   # (32, 768)
            img_feature = (image_features[:, 0])          # (32, 768)
            img_feature_proj = (image_features_proj[:, 0])   # (32, 512)


        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)    # (32,77)
            cls_cloth_score = self.classifier_cloth(feat)   # (32,256)
            cls_score_proj = self.classifier_proj(feat_proj)  # (32,77)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature,
                                                 img_feature_proj], img_feature_proj, cls_cloth_score, [stripe_features,stripe_features_proj]

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

