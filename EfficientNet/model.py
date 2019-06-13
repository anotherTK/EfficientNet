import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    Conv2dSamePadding,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        # BN的动量
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        # 是否启动SE模块
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        # 是否使用drop connect
        self.id_skip = block_args.id_skip

        # 扩张(expansion)阶段
        # 输入filter的数目
        inp = self._block_args.input_filters
        # 输出filter的数目
        oup = self._block_args.input_filters * self._block_args.expand_ratio

        if self._block_args.expand_ratio != 1:
            # 采用1x1卷积scale channel
            self._expand_conv = Conv2dSamePadding(
                in_channels=inp,
                out_channels=oup,
                kernel_size=1,
                bias=False,
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup,
                momentum=self._bn_mom,
                eps=self._bn_eps,
            )

        # 深度可分离卷积阶段
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2dSamePadding(
            in_channels=oup,
            out_channels=oup,
            # 深度可分离
            groups=oup,
            kernel_size=k,
            # 是否降低分辨率
            stride=s,
            bias=False,
        )

        self._bn1 = nn.BatchNorm2d(
            num_features=oup,
            momentum=self._bn_mom,
            eps=self._bn_eps,
        )

        # SE模块
        if self.has_se:
            # 确定squeeze分支的channel数量
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))

            self._se_reduce = Conv2dSamePadding(
                in_channels=oup,
                out_channels=num_squeezed_channels,
                kernel_size=1,
            )

            self._se_expand = Conv2dSamePadding(
                in_channels=num_squeezed_channels,
                out_channels=oup,
                kernel_size=1,
            )

        # 输出阶段
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2dSamePadding(
            in_channels=oup,
            out_channels=final_oup,
            kernel_size=1,
            bias=False,
        )

        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup,
            momentum=self._bn_mom,
            eps=self._bn_eps,
        )

    def forward(self, inputs, drop_connect_rate=None):

        x = inputs
        # Expansion
        if self._block_args.expand_ratio != 1:
            # 采用swise激活函数
            x = relu_fn(self._bn0(self._expand_conv(inputs)))

        # Depthwise
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # SE
        if self.has_se:
            # 全局均值池化
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            # reduce-expand
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            # integral/ re-weight
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # 默认是0.2
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            # 跳跃连接
            x = x + inputs
        return x


class EfficientNet(nn.Module):


    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # BN参数
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem 模块
        in_channels = 3 # rgb
        # 调整输出网络宽度
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2dSamePadding(
            in_channels, out_channels,
            kernel_size=3,
            # 分辨率下降
            stride=2,
            bias=False,
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=bn_mom,
            eps=bn_eps,
        )

        # 基本模块
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # 更新网络宽度和深度
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # 第一个block需要处理stride和channel
            self._blocks.append(MBConvBlock(block_args, self._global_params))

            # 然后repeat
            if block_args.num_repeat > 1:
                # 更新channel和stride
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        # 上一个阶段的输出channel数目
        in_channels = block_args.output_filters
        # 更新网络宽度
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2dSamePadding(
            in_channels, out_channels,
            kernel_size=1,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=bn_mom,
            eps=bn_eps
        )

        # 最后分类的线性网络
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # Loss函数
        self._loss = nn.CrossEntropyLoss()

    def extract_features(self, inputs):

        # Stem 模块
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # 基本模块
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            # TODO: 这个参数在这里好像没有作用？
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x)

        return x

    def forward(self, inputs, targets=None):

        x = self.extract_features(inputs)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        # 全局pooling
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        # dropout
        if self._dropout:
            # dropout需要给出是否是训练的标志
            x = F.dropout(x, p=self._dropout, training=self.training)
        # FC
        x = self._fc(x)

        # 判断是否训练
        if self.training:
            x = self._loss(x, targets)
        return x

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """目前官方只给了前四个预训练模型"""
        num_models = 4
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-', '_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name):
        model = EfficientNet.from_name(model_name)
        load_pretrained_weights(model, model_name)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _,_,res,_ = efficientnet_params(model_name)
        return res
