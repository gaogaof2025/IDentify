import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss_CMSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8
        self.factor = 0.2
        self.logit_scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, image_features, text_features, labels):
        """
        Similarity Distribution Matching
        """
        batch_size = image_features.shape[0]
        # pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
        # pid_dist = pid - pid.t()
        # labels = (pid_dist == 0).float()
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0).float()


        image_norm = image_features / image_features.norm(dim=1, keepdim=True)
        text_norm = text_features / text_features.norm(dim=1, keepdim=True)

        t2i_cosine_theta = text_norm @ image_norm.t()
        i2t_cosine_theta = t2i_cosine_theta.t()

        text_proj_image = self.logit_scale * t2i_cosine_theta
        image_proj_text = self.logit_scale * i2t_cosine_theta

        # normalize the true matching distribution
        # labels_distribute = labels / labels.sum(dim=1)
        labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return loss
