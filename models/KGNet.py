import torch
import torch.nn as nn
import torch.nn.functional as F
import os


from config.config import BaseConfig
from models.networks.extractors import resnet50
from models.networks.aux_decoder import Decoder
from models.networks.attri_attention import AttAttention



class KGNet(nn.Module):
    def __init__(self, parser):
        super(KGNet, self).__init__()
        self.parser = parser
        self.args = self.parser.get_args()

        self.n_instance = self.args.num_instance
        self.n_classes = self.args.num_classes
        self.embed_dim = self.args.embedding_dim
        self.branch_size = 256
        self.deep_features_size = 2048
        self.backend = 'resnet50'
        self.pretrained = self.args.pre_trained
        self.model_path = self.args.model_path
        self.final_dim = self.args.final_dim
        self.attri_dim = 5
        self.user_defined_embed = torch.Tensor([[2, 2, 2, 1, 1], # N1
                                                [2, 2, 2, 0, 2],
                                                [2, 1, 0, 0, 3],
                                                [2, 0, 0, 0, 4],
                                                [2, 2, 2, 2, 0]]).cuda() # W

        self.network_name()

        self.extractors = resnet50(self.pretrained, self.model_path)

        self.conv1 = nn.Sequential(
            self.extractors.conv1,
            self.extractors.bn1,
            self.extractors.relu
        )
        self.encoder2 = self.extractors.layer1  # 256
        self.encoder3 = self.extractors.layer2  # 512
        self.encoder4 = self.extractors.layer3  # 1024
        self.encoder5 = self.extractors.layer4  # 2048

        self.decoder4 = Decoder(2048 + 1024, 1024, 256)
        self.decoder3 = Decoder(256 + 512, 512, 256)
        self.decoder2 = Decoder(256 + 256, 256, 256)
        self.decoder1 = Decoder(256, 64, 256)

        self.logit1 = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )

        self.logit2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, self.final_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.final_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.deep_features_size + 1024 + self.final_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.n_classes),
        )

        self.embedding = nn.Sequential(
            nn.Conv2d(self.final_dim, self.embed_dim, 1)
        )
        # v2_2
        self.segmenting = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.n_instance, 1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.base_conv = nn.Sequential(
            nn.Conv2d(2048, 128, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # attribution embeding learning(zsl)
        self.attri_encoder = self.extractors.layer4  # 2048
        self.attri_attention = AttAttention(dim=2048, kernel_size=3)
        self.attri_embedder = nn.Conv2d(
            2048,
            self.attri_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.attri_avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)

    def forward(self, x):
        e1 = self.conv1(x)  # 64
        # print('e1:', e1.shape)
        e2 = self.encoder2(e1)  # 256
        # print('e2:', e2.shape)
        e3 = self.encoder3(e2)  # 512
        # print('e3:', e3.shape)
        e4 = self.encoder4(e3)
        # print('e4:', e4.shape)
        e5 = self.encoder5(e4)
        # print('e5:', e5.shape)

        d4 = self.decoder4(e5, e4)
        # print('d4:', d4.shape)
        d3 = self.decoder3(d4, e3)
        # print('d3:', d3.shape)
        d2 = self.decoder2(d3, e2)
        # print('d2:', d2.shape)
        d1 = self.decoder1(d2)
        # print('d1:', d1.shape)
        f = torch.cat((
            d1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=False),
        ), 1)
        # print('f:', f.shape)
        logit1 = self.logit1(f)
        logit2 = self.logit2(logit1)
        # print('logit:{}|{}'.format(logit1.shape, logit2.shape))

        # ------------------- ZSL ------------------- #
        attri_feat = self.attri_encoder(e4)
        attri_em = self.attri_avgpool(self.attri_embedder(attri_feat))
        # print(attri_em.shape)
        # print('att_feat:', att_feat.shape)
        e5 = self.attri_attention(e5, attri_feat)
        # print('e5 after att_feat:', e5.shape)
        attri_em = torch.flatten(attri_em, 1)
        sim_l = sematic_similarity(attri_em, self.user_defined_embed)

        # ------------------- C ------------------- #
        e_avg = self.avgpool(e5)

        new_e = self.S2C(f, e_avg, logit2)
        y_cla = self.classifier(torch.flatten(new_e, 1))
        # print('y_cla:{}'.format(y_cla.shape))

        # ------------------- S ------------------- #
        new_c = self.C2S(logit1, e_avg)
        # print('new_c:', new_c.shape)
        y_seg = self.segmenting(new_c)

        # ------------------- E ------------------- #
        y_em = self.embedding(logit2)

        return y_cla, y_seg, y_em, sim_l

    def C2S(self, S_out, C_out):
        C_out = self.base_conv(C_out)
        h, w = S_out.size()[2], S_out.size()[3]
        new_c = C_out.repeat(1, 1, h, w)
        return torch.mul(S_out, new_c)

    def S2C(self, S_out, C_out, E_out):
        return torch.cat([C_out, self.avgpool(S_out), self.avgpool(E_out)], dim=1)

    def network_name(self):
        return print('model type: gcnet+zsl.')


def sematic_similarity(attri_em, user_defined_embed):
    # 当输入是都是二维时，就是普通的矩阵乘法，和tensor.mm函数用法相同。
    # print('attri_em:{}, user_defined_embed:{}'.format(attri_em.shape, user_defined_embed.shape))
    # print('attri_em:', attri_em)
    scores_emb = torch.matmul(attri_em, user_defined_embed.T)
    # print(scores_emb)
    return scores_emb





if __name__ == '__main__':
    # e1: torch.Size([7, 64, 64, 128])
    # e2: torch.Size([7, 256, 64, 128])
    # e3: torch.Size([7, 512, 32, 64])
    # e4: torch.Size([7, 1024, 32, 64])
    # e5: torch.Size([7, 2048, 32, 64])
    # d4: torch.Size([7, 256, 32, 64])
    # d3: torch.Size([7, 256, 32, 64])
    # d2: torch.Size([7, 256, 64, 128])
    # d1: torch.Size([7, 256, 128, 256])
    # f: torch.Size([7, 1024, 128, 256])
    # logit:torch.Size([7, 128, 128, 256])|torch.Size([7, 8, 128, 256])
    parser = BaseConfig(
        os.path.join("../config/", "config.yaml"))
    # args = parser.get_args()

    net = GCNetZSL(parser).cuda()
    img = torch.rand([2, 3, 128, 256])
    img_gt = torch.Tensor([1, 2]).long().cuda()
    segLabel = torch.rand([2, 1, 128, 256])
    cla, seg, emb, sim_l = net(img.cuda())
    print(cla.shape, seg.shape, emb.shape)
    print(sim_l.shape, sim_l)

    # 计算zsl的损失
    criterion = nn.CrossEntropyLoss().cuda()
    print(sim_l.shape, img_gt.shape)  # torch.Size([2, 5]) torch.Size([2])
    loss = criterion(sim_l, img_gt)
    print(loss)
