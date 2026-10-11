import torch
import torch.nn as nn
import numpy as np
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
        # self.bottleneck_clo = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck_clo.bias.requires_grad_(False)
        # self.bottleneck_clo.apply(weights_init_kaiming)
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

        self.cam = CAM_Module(self.in_planes_proj)
        self.cam_1 = CAM_Module(self.in_planes)
        #
        self.sam = SpatialAttention()
        self.conv_c = nn.Sequential(
            nn.Conv2d(768 * 2, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        # G_o: 可见光匹配变换 (公式3)
        self.conv_o = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.conv_v = nn.Conv2d(768, 768, kernel_size=1)

        self.conv_c2 = nn.Sequential(
            nn.Conv2d(512 * 2, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # G_o: 可见光匹配变换 (公式3)
        self.conv_o2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_v2 = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, x=None, label=None, get_image=False, get_text=False, noise_img=None):
        if get_text == True:
            prompts, prompts_original = self.prompt_learner(label)
            text_features = (self.text_encoder(prompts, self.prompt_learner.tokenized_prompts))
            text_features_original = (
                self.text_encoder(prompts_original, self.prompt_learner.tokenized_prompts_original))
            return text_features, text_features_original

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
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)

            if noise_img is not None:
                noise_image_features_last, noise_image_features, noise_image_features_proj = self.image_encoder(
                    noise_img, cv_embed)

                #######################################################
                # 计算注意力权重
                attention_weights = torch.sigmoid(
                    torch.sum(image_features * noise_image_features, dim=-1, keepdim=True))

                # 动态调整特征
                feature_diff = image_features - noise_image_features
                feature_proj_diff = image_features_proj - noise_image_features_proj
                cloth_noise = attention_weights * feature_diff
                cloth_proj_noise = attention_weights * feature_proj_diff

                image_features = image_features - cloth_noise
                image_features_proj = image_features_proj - cloth_proj_noise
                ####################################################

            # image_features_proj = self.cam(image_features_proj)
            # image_features_proj = self.sam(image_features_proj)
            # image_features = self.cam_1(image_features)
            # image_features = self.sam(image_features)
            token = image_features[:, 0:1, :]
            stripe_features = image_features[:, 1:, :]
            f_vis = stripe_features.permute(0, 2, 1).contiguous().view(image_features.shape[0], 768, 24,
                                                                       12)  # (32,768,24,12)

            stripe_features_n = noise_image_features[:, 1:, :]  # (16,288,768)
            f_ir = stripe_features_n.permute(0, 2, 1).contiguous().view(image_features.shape[0], 768, 24,
                                                                        12)  # (32,768,24,12)
            f_c_prime = self.conv_c(torch.cat([f_vis, f_ir], dim=1))  # (32,768,24,12)
            S_v_c = F.softmax(self.conv_o(f_vis) * f_c_prime, dim=1)  # (32,768,24,12)
            # 公式(5-6): 特征增强（残差连接）
            f_vis_global = f_vis + self.conv_v(f_c_prime * S_v_c)  # (32,768,24,12)
            f_vis_global = f_vis_global.view(image_features.shape[0], 768, -1).permute(0, 2, 1)  # (32, 288, 768)
            image_features = torch.cat([token, f_vis_global], dim=1)  # (32, 289, 768)

            token2 = image_features_proj[:, 0:1, :]
            stripe_features2 = image_features_proj[:, 1:, :]
            f_vis2 = stripe_features2.permute(0, 2, 1).contiguous().view(image_features_proj.shape[0], 512, 24,
                                                                         12)  # (32,768,24,12)
            stripe_features_2 = noise_image_features_proj[:, 1:, :]
            f_ir2 = stripe_features_2.permute(0, 2, 1).contiguous().view(image_features_proj.shape[0], 512, 24,
                                                                         12)  # (32,768,24,12)
            f_c_prime2 = self.conv_c2(torch.cat([f_vis2, f_ir2], dim=1))
            S_v_c2 = F.softmax(self.conv_o2(f_vis2) * f_c_prime2, dim=1)
            # 公式(5-6): 特征增强（残差连接）
            f_vis_global2 = f_vis2 + self.conv_v2(f_c_prime2 * S_v_c2)
            f_vis_global2 = f_vis_global2.view(image_features_proj.shape[0], 512, -1).permute(0, 2, 1)  # (32, 288, 768)
            image_features_proj = torch.cat([token2, f_vis_global2], dim=1)  # (32, 289, 768)

            img_feature_last = image_features_last[:, 0]
            img_feature = (image_features[:, 0])
            img_feature_proj = (image_features_proj[:, 0])

        feat = self.bottleneck(img_feature)
        feat_clo = self.bottleneck_clo(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_cloth_score = self.classifier_cloth(feat_clo)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature,
                                                 img_feature_proj], img_feature_proj, cls_cloth_score

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
        # === 掩码图像的文本提示模板 ===
        ctx_init = "A photo of a X X X X person."
        # ctx_init = "A photo of a X X X X person with X X cloth."
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

        # === 原始图像的文本提示模板 ===
        ctx_init_original = "A photo of a X X X X person."
        ctx_init_original = ctx_init_original.replace("_", " ")

        tokenized_prompts_original = clip.tokenize(ctx_init_original).cuda()
        with torch.no_grad():
            embedding_original = token_embedding(tokenized_prompts_original).type(dtype)
        self.tokenized_prompts_original = tokenized_prompts_original

        # 原始图像对应的可学习参数
        cls_vectors_original = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors_original, std=0.02)
        self.cls_ctx_original = nn.Parameter(cls_vectors_original)  # 原始图像的文本向量

        self.token_prefix_original = embedding_original[:, :n_ctx + 1, :]
        self.token_suffix_original = embedding_original[:, n_ctx + 1 + n_cls_ctx:, :]

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        # cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        cls_ctx = self.cls_ctx[label]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1
        )
        cls_ctx_original = self.cls_ctx_original[label]
        prefix_original = self.token_prefix_original.expand(b, -1, -1)
        suffix_original = self.token_suffix_original.expand(b, -1, -1)
        prompts_original = torch.cat(
            [prefix_original, cls_ctx_original, suffix_original],
            dim=1)

        return prompts, prompts_original