import numpy as np
import torch
from torch import nn
import torch.functional as F
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .ctnet_head import CtnetHead
from .conv_rnn import CLSTM_cell


def parse_dynamic_params(params,
                         channels,
                         weight_nums,
                         bias_nums,
                         out_channels=1,
                         mask=True):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    # params: (num_ins x n_param)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]
    if mask:
        bias_splits[-1] = bias_splits[-1] - 2.19

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_insts * out_channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * out_channels)

    return weight_splits, bias_splits


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(
        0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class DynamicMaskHead(nn.Module):

    def __init__(self,
                 num_layers,
                 channels,
                 in_channels,
                 mask_out_stride,
                 weight_nums,
                 bias_nums,
                 disable_coords=False,
                 shape=(160, 256),
                 out_channels=1,
                 compute_locations_pre=True,
                 location_configs=None):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.in_channels = in_channels
        self.mask_out_stride = mask_out_stride
        self.disable_coords = disable_coords

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.out_channels = out_channels
        self.compute_locations_pre = compute_locations_pre
        self.location_configs = location_configs

        if compute_locations_pre and location_configs is not None:
            N, _, H, W = location_configs['size']
            device = location_configs['device']
            locations = compute_locations(H, W, stride=1, device='cpu')

            locations = locations.unsqueeze(0).permute(
                0, 2, 1).contiguous().float().view(1, 2, H, W)
            locations[:0, :, :] /= H
            locations[:1, :, :] /= W
            locations = locations.repeat(N, 1, 1, 1)
            self.locations = locations.to(device)

    def forward(self, x, mask_head_params, num_ins, is_mask=True):

        N, _, H, W = x.size()
        if not self.disable_coords:
            if self.compute_locations_pre and self.location_configs is not None:
                locations = self.locations.to(x.device)
            else:
                locations = compute_locations(
                    x.size(2), x.size(3), stride=1, device='cpu')
                locations = locations.unsqueeze(0).permute(
                    0, 2, 1).contiguous().float().view(1, 2, H, W)
                locations[:0, :, :] /= H
                locations[:1, :, :] /= W
                locations = locations.repeat(N, 1, 1, 1)
                locations = locations.to(x.device)

            #relative_coords = relative_coords.to(dtype=mask_feats.dtype)
            x = torch.cat([locations, x], dim=1)
        mask_head_inputs = []
        for idx in range(N):
            mask_head_inputs.append(x[idx:idx + 1, ...].repeat(
                1, num_ins[idx], 1, 1))
        mask_head_inputs = torch.cat(mask_head_inputs, 1)
        num_insts = sum(num_ins)
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)
        weights, biases = parse_dynamic_params(
            mask_head_params,
            self.channels,
            self.weight_nums,
            self.bias_nums,
            out_channels=self.out_channels,
            mask=is_mask)
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights,
                                              biases, num_insts)
        mask_logits = mask_logits.view(1, -1, H, W)
        return mask_logits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(n, k, 1)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module
class CondLaneHead(nn.Module):

    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 head_channels=64,
                 head_layers=1,
                 disable_coords=False,
                 branch_in_channels=288,
                 branch_channels=64,
                 branch_out_channels=64,
                 reg_branch_channels=32,
                 branch_num_conv=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 hm_idx=-1,
                 mask_idx=0,
                 compute_locations_pre=True,
                 location_configs=None,
                 mask_norm_act=True,
                 regression=True,
                 train_cfg=None,
                 test_cfg=None):
        super(CondLaneHead, self).__init__()
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.mask_idx = mask_idx
        self.regression = regression
        if mask_norm_act:
            final_norm_cfg = dict(type='BN', requires_grad=True)
            final_act_cfg = dict(type='ReLU')
        else:
            final_norm_cfg = None
            final_act_cfg = None
        # mask branch
        mask_branch = []
        mask_branch.append(
            ConvModule(
                sum(in_channels),
                branch_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg))
        for i in range(branch_num_conv):
            mask_branch.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg))
        mask_branch.append(
            ConvModule(
                branch_channels,
                branch_out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=final_norm_cfg,
                act_cfg=final_act_cfg))
        self.add_module('mask_branch', nn.Sequential(*mask_branch))

        self.mask_weight_nums, self.mask_bias_nums = self.cal_num_params(
            head_layers, disable_coords, head_channels, out_channels=1)

        self.num_mask_params = sum(self.mask_weight_nums) + sum(
            self.mask_bias_nums)

        self.reg_weight_nums, self.reg_bias_nums = self.cal_num_params(
            head_layers, disable_coords, head_channels, out_channels=1)

        self.num_reg_params = sum(self.reg_weight_nums) + sum(
            self.reg_bias_nums)
        if self.regression:
            self.num_gen_params = self.num_mask_params + self.num_reg_params
        else:
            self.num_gen_params = self.num_mask_params
            self.num_reg_params = 0

        self.mask_head = DynamicMaskHead(
            head_layers,
            branch_out_channels,
            branch_out_channels,
            1,
            self.mask_weight_nums,
            self.mask_bias_nums,
            disable_coords=False,
            compute_locations_pre=compute_locations_pre,
            location_configs=location_configs)
        if self.regression:
            self.reg_head = DynamicMaskHead(
                head_layers,
                branch_out_channels,
                branch_out_channels,
                1,
                self.reg_weight_nums,
                self.reg_bias_nums,
                disable_coords=False,
                out_channels=1,
                compute_locations_pre=compute_locations_pre,
                location_configs=location_configs)
        if 'params' not in heads:
            heads['params'] = num_classes * (
                self.num_mask_params + self.num_reg_params)

        self.ctnet_head = CtnetHead(
            heads,
            channels_in=branch_in_channels,
            final_kernel=1,
            # head_conv=64,)
            head_conv=branch_in_channels)

        self.feat_width = location_configs['size'][-1]
        self.mlp = MLP(self.feat_width, 64, 2, 2)

    def cal_num_params(self,
                       num_layers,
                       disable_coords,
                       channels,
                       out_channels=1):
        weight_nums, bias_nums = [], []
        for l in range(num_layers):
            if l == num_layers - 1:
                if num_layers == 1:
                    weight_nums.append((channels + 2) * out_channels)
                else:
                    weight_nums.append(channels * out_channels)
                bias_nums.append(out_channels)
            elif l == 0:
                if not disable_coords:
                    weight_nums.append((channels + 2) * channels)
                else:
                    weight_nums.append(channels * channels)
                bias_nums.append(channels)

            else:
                weight_nums.append(channels * channels)
                bias_nums.append(channels)
        return weight_nums, bias_nums

    def ctdet_decode(self, heat, thr=0.1):

        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2

            hmax = nn.functional.max_pool2d(
                heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()
            return heat * keep

        def _format(heat, inds):
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                ret.append({
                    'coord': coord,
                    'id_class': id_class,
                    'score': score
                })
            return ret

        heat_nms = _nms(heat)
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        inds = np.where(heat_nms > thr)
        seeds = _format(heat_nms, inds)
        return seeds

    def forward_train(self, inputs, pos, num_ins):
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]

        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.size()[0]

        # f_mask
        z = self.ctnet_head(f_hm)
        hm, params = z['hm'], z['params']
        h_hm, w_hm = hm.size()[2:]
        h_mask, w_mask = f_mask.size()[2:]
        params = params.view(m_batchsize, self.num_classes, -1, h_hm, w_hm)
        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch
        # reg_branch = self.reg_branch(f_mask)
        params = params.permute(0, 1, 3, 4,
                                2).contiguous().view(-1, self.num_gen_params)

        pos_tensor = torch.from_numpy(np.array(pos)).long().to(
            params.device).unsqueeze(1)

        pos_tensor = pos_tensor.expand(-1, self.num_gen_params)
        mask_pos_tensor = pos_tensor[:, :self.num_mask_params]
        reg_pos_tensor = pos_tensor[:, self.num_mask_params:]
        if pos_tensor.size()[0] == 0:
            masks = None
            feat_range = None
        else:
            mask_params = params[:, :self.num_mask_params].gather(
                0, mask_pos_tensor)
            masks = self.mask_head(mask_branch, mask_params, num_ins)
            if self.regression:
                reg_params = params[:, self.num_mask_params:].gather(
                    0, reg_pos_tensor)
                regs = self.reg_head(reg_branch, reg_params, num_ins)
            else:
                regs = masks
            # regs = regs.view(sum(num_ins), 1, h_mask, w_mask)
            feat_range = masks.permute(0, 1, 3,
                                       2).view(sum(num_ins), w_mask, h_mask)
            feat_range = self.mlp(feat_range)
        return hm, regs, masks, feat_range, [mask_branch, reg_branch]

    def forward_test(
            self,
            inputs,
            hack_seeds=None,
            hm_thr=0.3,
    ):

        def parse_pos(seeds, batchsize, num_classes, h, w, device):
            pos_list = [[p['coord'], p['id_class'] - 1] for p in seeds]
            poses = []
            for p in pos_list:
                [c, r], label = p
                pos = label * h * w + r * w + c
                poses.append(pos)
            poses = torch.from_numpy(np.array(
                poses, np.long)).long().to(device).unsqueeze(1)
            return poses

        # with Timer("Elapsed time in stage1: %f"):  # ignore
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.size()[0]
        f_deep = f_mask
        m_batchsize = f_deep.size()[0]
        # with Timer("Elapsed time in ctnet_head: %f"):  # 0.3ms
        z = self.ctnet_head(f_hm)
        h_hm, w_hm = f_hm.size()[2:]
        h_mask, w_mask = f_mask.size()[2:]
        hm, params = z['hm'], z['params']
        hm = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)
        params = params.view(m_batchsize, self.num_classes, -1, h_hm, w_hm)
        # with Timer("Elapsed time in two branch: %f"):  # 0.6ms
        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch
        # reg_branch = self.reg_branch(f_mask)
        params = params.permute(0, 1, 3, 4,
                                2).contiguous().view(-1, self.num_gen_params)

        batch_size, num_classes, h, w = hm.size()
        # with Timer("Elapsed time in ct decode: %f"):  # 0.2ms
        seeds = self.ctdet_decode(hm, thr=hm_thr)
        if hack_seeds is not None:
            seeds = hack_seeds
        # with Timer("Elapsed time in stage2: %f"):  # 0.08ms
        pos_tensor = parse_pos(seeds, batch_size, num_classes, h, w, hm.device)
        pos_tensor = pos_tensor.expand(-1, self.num_gen_params)
        num_ins = [pos_tensor.size()[0]]
        mask_pos_tensor = pos_tensor[:, :self.num_mask_params]
        if self.regression:
            reg_pos_tensor = pos_tensor[:, self.num_mask_params:]
        # with Timer("Elapsed time in stage3: %f"):  # 0.8ms
        if pos_tensor.size()[0] == 0:
            return [], hm
        else:
            mask_params = params[:, :self.num_mask_params].gather(
                0, mask_pos_tensor)
            # with Timer("Elapsed time in mask_head: %f"):  #0.3ms
            masks = self.mask_head(mask_branch, mask_params, num_ins)
            if self.regression:
                reg_params = params[:, self.num_mask_params:].gather(
                    0, reg_pos_tensor)
                # with Timer("Elapsed time in reg_head: %f"):  # 0.25ms
                regs = self.reg_head(reg_branch, reg_params, num_ins)
            else:
                regs = masks
            feat_range = masks.permute(0, 1, 3,
                                       2).view(sum(num_ins), w_mask, h_mask)
            feat_range = self.mlp(feat_range)
            for i in range(len(seeds)):
                seeds[i]['reg'] = regs[0, i:i + 1, :, :]
                m = masks[0, i:i + 1, :, :]
                seeds[i]['mask'] = m
                seeds[i]['range'] = feat_range[i:i + 1]
            return seeds, hm

    def inference_mask(self, pos):
        pass

    def forward(
            self,
            x_list,
            hm_thr=0.3,
    ):
        return self.forward_test(x_list, )

    def init_weights(self):
        # ctnet_head will init weights during building
        pass


class PredictFC(nn.Module):

    def __init__(self, num_params, num_states, in_channels):
        super(PredictFC, self).__init__()
        self.num_params = num_params
        self.fc_param = nn.Conv2d(
            in_channels,
            num_params,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc_state = nn.Conv2d(
            in_channels,
            num_states,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, input):
        params = self.fc_param(input)
        state = self.fc_state(input)
        return params, state


@HEADS.register_module
class CondLaneRNNHead(nn.Module):

    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 ct_head,
                 head_channels=64,
                 head_layers=1,
                 disable_coords=False,
                 branch_channels=64,
                 branch_out_channels=64,
                 reg_branch_channels=32,
                 branch_num_conv=1,
                 num_params=256,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 hm_idx=-1,
                 mask_idx=0,
                 compute_locations_pre=True,
                 location_configs=None,
                 zero_hidden_state=False,
                 train_cfg=None,
                 test_cfg=None):
        super(CondLaneRNNHead, self).__init__()
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.mask_idx = mask_idx
        self.zero_hidden_state = zero_hidden_state

        # mask branch
        mask_branch = []
        mask_branch.append(
            ConvModule(
                sum(in_channels),
                branch_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=norm_cfg))
        for i in range(branch_num_conv):
            mask_branch.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg))
        mask_branch.append(
            ConvModule(
                branch_channels,
                branch_out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=None,
                act_cfg=None))
        self.add_module('mask_branch', nn.Sequential(*mask_branch))

        self.mask_weight_nums, self.mask_bias_nums = self.cal_num_params(
            head_layers, disable_coords, branch_out_channels, out_channels=1)

        self.num_mask_params = sum(self.mask_weight_nums) + sum(
            self.mask_bias_nums)

        self.reg_weight_nums, self.reg_bias_nums = self.cal_num_params(
            head_layers, disable_coords, reg_branch_channels, out_channels=1)

        self.num_reg_params = sum(self.reg_weight_nums) + sum(
            self.reg_bias_nums)
        self.num_gen_params = self.num_mask_params + self.num_reg_params

        self.mask_head = DynamicMaskHead(
            head_layers,
            branch_out_channels,
            branch_out_channels,
            1,
            self.mask_weight_nums,
            self.mask_bias_nums,
            disable_coords=False,
            compute_locations_pre=compute_locations_pre,
            location_configs=location_configs)
        self.reg_head = DynamicMaskHead(
            head_layers,
            reg_branch_channels,
            reg_branch_channels,
            1,
            self.reg_weight_nums,
            self.reg_bias_nums,
            disable_coords=False,
            out_channels=1,
            compute_locations_pre=compute_locations_pre,
            location_configs=location_configs)
        self.ctnet_head = CtnetHead(
            ct_head['heads'],
            channels_in=ct_head['channels_in'],
            final_kernel=ct_head['final_kernel'],
            head_conv=ct_head['head_conv'])
        self.rnn_in_channels = ct_head['heads']['params']
        self.rnn_ceil = CLSTM_cell((1, 1), self.rnn_in_channels, 1,
                                   self.rnn_in_channels)
        self.final_fc = PredictFC(self.num_gen_params, 2, self.rnn_in_channels)
        self.feat_width = location_configs['size'][-1]
        self.mlp = MLP(self.feat_width, 64, 2, 2)

    def cal_num_params(self,
                       num_layers,
                       disable_coords,
                       channels,
                       out_channels=1):
        weight_nums, bias_nums = [], []
        for l in range(num_layers):
            if l == num_layers - 1:
                if num_layers == 1:
                    weight_nums.append((channels + 2) * out_channels)
                else:
                    weight_nums.append(channels * out_channels)
                bias_nums.append(out_channels)
            elif l == 0:
                if not disable_coords:
                    weight_nums.append((channels + 2) * channels)
                else:
                    weight_nums.append(channels * channels)
                bias_nums.append(channels)

            else:
                weight_nums.append(channels * channels)
                bias_nums.append(channels)
        return weight_nums, bias_nums

    def ctdet_decode(self, heat, thr=0.1):

        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2

            hmax = nn.functional.max_pool2d(
                heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()
            return heat * keep

        def _format(heat, inds):
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                ret.append({
                    'coord': coord,
                    'id_class': id_class,
                    'score': score
                })
            return ret

        heat_nms = _nms(heat)
        heat_nms = heat_nms.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        inds = np.where(heat_nms > thr)
        seeds = _format(heat_nms, inds)
        return seeds

    def forward_train(self, inputs, pos, num_ins, memory):

        def choose_idx(num_ins, idx):
            count = 0
            for i in range(len(num_ins) - 1):
                if idx >= count and idx < count + num_ins[i]:
                    return i
                count += num_ins[i]
            return len(num_ins) - 1

        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]

        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.size()[0]

        # f_mask
        z = self.ctnet_head(f_hm)
        hm, params = z['hm'], z['params']
        h_hm, w_hm = hm.size()[2:]
        h_mask, w_mask = f_mask.size()[2:]
        params = params.view(m_batchsize, self.num_classes, -1, h_hm, w_hm)
        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch
        params = params.permute(0, 1, 3, 4,
                                2).contiguous().view(-1, self.rnn_in_channels)
        pos_array = np.array([p[0] for p in pos], np.int32)
        pos_tensor = torch.from_numpy(pos_array).long().to(
            params.device).unsqueeze(1)

        pos_tensor = pos_tensor.expand(-1, self.rnn_in_channels)
        states = []
        kernel_params = []
        if pos_tensor.size()[0] == 0:
            masks = None
            regs = None
        else:
            num_ins_per_seed = []
            rnn_params = params.gather(0, pos_tensor)
            ins_count = 0
            for idx, (_, r_times) in enumerate(pos):
                rnn_feat_input = rnn_params[idx:idx + 1, :]
                rnn_feat_input = rnn_feat_input.reshape(1, -1, 1, 1)
                hidden_h = rnn_feat_input
                hidden_c = rnn_feat_input
                rnn_feat_input = rnn_feat_input.reshape(1, 1, -1, 1, 1)
                if self.zero_hidden_state:
                    hidden_state = None
                else:
                    hidden_state = (hidden_h, hidden_c)
                num_ins_count = 0
                for _ in range(r_times):
                    rnn_out, hidden_state = self.rnn_ceil(
                        inputs=rnn_feat_input,
                        hidden_state=hidden_state,
                        seq_len=1)
                    rnn_out = rnn_out.reshape(1, -1, 1, 1)
                    k_param, state = self.final_fc(rnn_out)
                    k_param = k_param.squeeze(-1).squeeze(-1)
                    state = state.squeeze(-1).squeeze(-1)
                    states.append(state)
                    kernel_params.append(k_param)
                    num_ins_count += 1
                    rnn_feat_input = rnn_out
                    rnn_feat_input = rnn_feat_input.reshape(1, 1, -1, 1, 1)
                    ins_count += 1

                num_ins_per_seed.append(num_ins_count)
            kernel_params = torch.cat(kernel_params, 0)
            states = torch.cat(states, 0)
            mask_params = kernel_params[:, :self.num_mask_params]
            reg_params = kernel_params[:, self.num_mask_params:]
            masks = self.mask_head(mask_branch, mask_params, num_ins)
            regs = self.reg_head(reg_branch, reg_params, num_ins)
            feat_range = masks.permute(0, 1, 3,
                                       2).view(sum(num_ins), w_mask, h_mask)
            feat_range = self.mlp(feat_range)

        return hm, regs, masks, feat_range, states

    def forward_test(
            self,
            inputs,
            hm_thr=0.3,
            max_rtimes=6,
            memory=None,
            hack_seeds=None,
    ):

        def parse_pos(seeds, batchsize, num_classes, h, w, device):
            pos_list = [[p['coord'], p['id_class'] - 1] for p in seeds]
            poses = []
            for p in pos_list:
                [c, r], label = p
                pos = label * h * w + r * w + c
                poses.append(pos)
            poses = torch.from_numpy(np.array(
                poses, np.long)).long().to(device).unsqueeze(1)
            return poses

        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.size()[0]
        f_deep = f_mask
        m_batchsize = f_deep.size()[0]

        z = self.ctnet_head(f_hm)
        h_hm, w_hm = f_hm.size()[2:]
        hm, params = z['hm'], z['params']
        hm = torch.clamp(hm.sigmoid(), min=1e-4, max=1 - 1e-4)
        h_mask, w_mask = f_mask.size()[2:]
        params = params.view(m_batchsize, self.num_classes, -1, h_hm, w_hm)

        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch
        self.debug_mask_branch = mask_branch
        self.debug_reg_branch = reg_branch
        params = params.permute(0, 1, 3, 4,
                                2).contiguous().view(-1, self.rnn_in_channels)

        batch_size, num_classes, h, w = hm.size()
        seeds = self.ctdet_decode(hm, thr=hm_thr)
        if hack_seeds is not None:
            seeds = hack_seeds
        pos_tensor = parse_pos(seeds, batch_size, num_classes, h, w, hm.device)
        pos_tensor = pos_tensor.expand(-1, self.rnn_in_channels)

        if pos_tensor.size()[0] == 0:
            return [], hm
        else:
            kernel_params = []
            num_ins_per_seed = []
            rnn_params = params.gather(0, pos_tensor)
            for idx in range(pos_tensor.size()[0]):
                rnn_feat_input = rnn_params[idx:idx + 1, :]
                rnn_feat_input = rnn_feat_input.reshape(1, -1, 1, 1)
                hidden_h = rnn_feat_input
                hidden_c = rnn_feat_input
                rnn_feat_input = rnn_feat_input.reshape(1, 1, -1, 1, 1)

                if self.zero_hidden_state:
                    hidden_state = None
                else:
                    hidden_state = (hidden_h, hidden_c)
                num_ins_count = 0
                for _ in range(max_rtimes):
                    rnn_out, hidden_state = self.rnn_ceil(
                        inputs=rnn_feat_input,
                        hidden_state=hidden_state,
                        seq_len=1)
                    rnn_out = rnn_out.reshape(1, -1, 1, 1)
                    k_param, state = self.final_fc(rnn_out)
                    k_param = k_param.squeeze(-1).squeeze(-1)
                    state = state.squeeze(-1).squeeze(-1)
                    kernel_params.append(k_param)
                    num_ins_count += 1
                    if torch.argmax(state[0]) == 0:
                        break
                    rnn_feat_input = rnn_out
                    rnn_feat_input = rnn_feat_input.reshape(1, 1, -1, 1, 1)
                num_ins_per_seed.append(num_ins_count)

            num_ins = len(kernel_params)
            kernel_params = torch.cat(kernel_params, 0)
            mask_params = kernel_params[:, :self.num_mask_params]
            reg_params = kernel_params[:, self.num_mask_params:]
            masks = self.mask_head(mask_branch, mask_params, [num_ins])
            regs = self.reg_head(reg_branch, reg_params, [num_ins])
            feat_range = masks.permute(0, 1, 3,
                                       2).view(num_ins, w_mask, h_mask)
            feat_range = self.mlp(feat_range)
            start_ins_idx = 0
            for i, idx_ins in enumerate(num_ins_per_seed):
                end_ins_idx = start_ins_idx + idx_ins
                seeds[i]['reg'] = regs[0, start_ins_idx:end_ins_idx, :, :]
                seeds[i]['mask'] = masks[0, start_ins_idx:end_ins_idx, :, :]
                seeds[i]['range'] = feat_range[start_ins_idx:end_ins_idx]
                start_ins_idx = end_ins_idx
        return seeds, hm

    def forward(
            self,
            x_list,
            hm_thr=0.3,
    ):
        return self.forward_test(x_list, )

    def init_weights(self):
        pass