import re
import math
import collections
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

# 创建一个全局参数的命名组
GlobalParams = collections.namedtuple('GlobalParams', ['batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor', 'min_depth', 'drop_connect_rate'])

# 创建一个基本模块的命名组
BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters', 'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# 给命名组设置默认值
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def relu_fn(x):
    """Swish激活函数
    
    Arguments:
        x {Tensor} -- 输入

    Returns:
        tensor -- 激活函数输出
    """
    return x * torch.sigmoid(x)

def round_filters(filters, global_params):
    """重新调整filter的数目（网络宽度）
    
    Arguments:
        filters {int} -- 输入的filter数量
        global_params {namedtuple} -- 全局参数
    
    Returns:
        int -- 调整后的filter数目
    """
    # 宽度系数
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    # 宽度增加
    filters *= multiplier
    # 判断前一个是否为真，为真则取其值
    min_depth = min_depth or divisor
    # 选取的filter数目必须能够整除divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # 如果新的filter数目太少了，那么就直接加上divisor
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """调整block的重复次数，网络深度
    
    Arguments:
        repeats {int} -- 重复次数
        global_params {namedtuple} -- 全局参数
    
    Returns:
        int -- 调整后的重复次数
    """
    # 深度系数
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """dropout 操作, 针对的是sample. TODO: 数据增强?
    
    Arguments:
        inputs {tensor} -- 输入向量
        p {float} -- 丢弃概率
        training {bool} -- 是否为训练过程的标志位
    
    Returns:
        tensor -- 丢弃+加权后输出
    """
    # 只有训练过程执行操作
    if not training:
        return inputs

    batch_size = inputs.shape[0]
    # 保持的概率
    keep_prob = 1 - p
    random_tensor = keep_prob
    # 对每个sample随机生成一个概率
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Conv2dSamePadding(nn.Conv2d):
    """继承自Conv2d，根据输入feature map大小和stride直接决定输出feature map大小，自适应padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        # 采用padding=0初始化父类
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        # stride是一个长度为2的list，[stride_x, stride_y]
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        # 输入feature map的大小
        ih, iw = x.size()[-2:]
        # 权重的size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        # 确定输出feature map的大小
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        # 确定需要padding的大小
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1)* self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1)* self.dilation[1] + 1 - iw, 0)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        

def efficientnet_params(model_name):
    """根据模型选择对应的参数设置
    
    Arguments:
        model_name {str} -- 模型名字
    
    Returns:
        tuple -- 参数元组
    """
    params_dict = {
        # coefficients: width, depth, res, dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }

    return params_dict[model_name]


class BlockDecoder(object):

    @staticmethod
    def _encode_block_string(block):
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]

        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ration)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        assert (('s' in options and len(options['s']) == 1) or (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])]
        )

    @staticmethod
    def encode(block_args):
        block_strings = []
        for block in block_args:
            block_strings.append(
                BlockDecoder._encode_block_string(block)
            )
        return block_strings

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        block_args = []
        for block_string in string_list:
            block_args.append(
                BlockDecoder._decode_block_string(block_string)
            )
        return block_args


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, drop_connect_rate=0.2):
    # 模型的配置
    # 这个是baseline模型的配置，具体的模型还需要根据额外的设置给没项进行增益
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    # 返回模型的具体配置
    if model_name.startswith('efficientnet'):
        # width, depth, res, dropout
        w, d, _, p = efficientnet_params(model_name)
        # 注意： 所有模型都有drop_connect_rate=0.2
        block_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)

    if override_params:
        global_params = global_params._replace(**override_params)
    
    return block_args, global_params


# 预训练模型链接/已经转换好的pytorch模型
url_map = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
}


def load_pretrained_weights(model, model_name):
    state_dict = model_zoo.load_url(url_map[model_name])
    model.load_state_dict(state_dict)
    print('Loaded pretrained weights for {}'.format(model_name))