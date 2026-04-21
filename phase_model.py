from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM1D(nn.Module):

    def __init__(self, channels: int, reduction: int=8, spatial_kernel: int=7):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_fc = nn.Sequential(nn.Conv1d(channels, hidden, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv1d(hidden, channels, kernel_size=1, bias=False))
        self.temporal_conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.register_buffer('last_temporal_att', torch.zeros(1, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=-1, keepdim=True)
        max_pool, _ = torch.max(x, dim=-1, keepdim=True)
        ca = torch.sigmoid(self.channel_fc(avg_pool) + self.channel_fc(max_pool))
        x_channel = x * ca
        avg_t = torch.mean(x_channel, dim=1, keepdim=True)
        max_t, _ = torch.max(x_channel, dim=1, keepdim=True)
        temporal_att = torch.sigmoid(self.temporal_conv(torch.cat([avg_t, max_t], dim=1)))
        self.last_temporal_att = temporal_att
        x_out = x_channel * temporal_att
        return x_out

class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, k: int, dropout: float=0.0):
        super().__init__()
        pad = k // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class ASFF1D(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight_conv = nn.Conv1d(channels * 2, 2, kernel_size=1, bias=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if x1.shape[-1] != x2.shape[-1]:
            target_T = min(x1.shape[-1], x2.shape[-1])
            if x1.shape[-1] != target_T:
                x1 = F.interpolate(x1, size=target_T, mode='linear', align_corners=False)
            if x2.shape[-1] != target_T:
                x2 = F.interpolate(x2, size=target_T, mode='linear', align_corners=False)
        w = torch.sigmoid(self.weight_conv(torch.cat([x1, x2], dim=1)))
        w1 = w[:, 0:1]
        w2 = w[:, 1:2]
        return x1 * w1 + x2 * w2

class TemporalBiFPNASFF(nn.Module):

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        if len(channels) < 3:
            raise ValueError('TemporalBiFPNASFF expects at least 3 scale features.')
        c1, c2, c3, c4, *rest = channels
        c5 = rest[0] if rest else c4
        self.conv4 = DepthwiseSeparableConv1d(c5 + c4, c4, k=3, dropout=0.0)
        self.conv3 = DepthwiseSeparableConv1d(c4 + c3, c3, k=3, dropout=0.0)
        self.conv2 = DepthwiseSeparableConv1d(c3 + c2, c2, k=3, dropout=0.0)
        self.conv1 = DepthwiseSeparableConv1d(c2 + c1, c1, k=3, dropout=0.0)
        self.asff4 = ASFF1D(c4)
        self.asff3 = ASFF1D(c3)
        self.asff2 = ASFF1D(c2)
        self.asff1 = ASFF1D(c1)

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == ref.shape[-1]:
            return x
        return F.interpolate(x, size=ref.shape[-1], mode='linear', align_corners=False)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p4_up = self._upsample_to(p5, p4)
        f4_mid = self.conv4(torch.cat([p4, p4_up], dim=1))
        f4 = self.asff4(p4, f4_mid)
        p3_up = self._upsample_to(f4, p3)
        f3_mid = self.conv3(torch.cat([p3, p3_up], dim=1))
        f3 = self.asff3(p3, f3_mid)
        p2_up = self._upsample_to(f3, p2)
        f2_mid = self.conv2(torch.cat([p2, p2_up], dim=1))
        f2 = self.asff2(p2, f2_mid)
        p1_up = self._upsample_to(f2, p1)
        f1_mid = self.conv1(torch.cat([p1, p1_up], dim=1))
        f1 = self.asff1(p1, f1_mid)
        f5 = p5
        return (f1, f2, f3, f4, f5)

class MultiScaleConv1d(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernels=(7,), dropout: float=0.0, use_separable: bool=False, use_cbam: bool=False, fusion_mode: str='concat', fusion_gate_hidden: int=16, fusion_residual_scale: float=0.3, fusion_use_maxpool: bool=True, cbam_modulate_softgate: bool=True, cbam_softgate_strength: float=1.0):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.fusion_residual_scale = fusion_residual_scale
        self.fusion_use_maxpool = fusion_use_maxpool
        self.cbam_modulate_softgate = cbam_modulate_softgate
        self.cbam_softgate_strength = float(cbam_softgate_strength)
        branch_cls = DepthwiseSeparableConv1d if use_separable else None
        branches = []
        for k in kernels:
            if branch_cls is None:
                layers: list[nn.Module] = [nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
                if dropout > 0.0:
                    layers.append(nn.Dropout(p=dropout))
                branches.append(nn.Sequential(*layers))
            else:
                branches.append(DepthwiseSeparableConv1d(in_ch, out_ch, k, dropout=dropout))
        self.branches = nn.ModuleList(branches)
        n_scales = len(kernels)
        self.fuse = nn.Sequential(nn.Conv1d(out_ch * n_scales, out_ch, kernel_size=1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))
        self.cbam = CBAM1D(out_ch, reduction=8, spatial_kernel=7) if use_cbam else nn.Identity()
        if fusion_mode == 'softgate_residual':
            gate_in_ch = out_ch * (2 if fusion_use_maxpool else 1)
            hidden = max(4, fusion_gate_hidden)
            self.scale_gates = nn.ModuleList([nn.Sequential(nn.Conv1d(gate_in_ch, hidden, kernel_size=1, bias=True), nn.ReLU(inplace=True), nn.Conv1d(hidden, out_ch, kernel_size=1, bias=True)) for _ in kernels])
            self.gated_fuse = nn.Sequential(nn.Conv1d(out_ch * n_scales, out_ch, kernel_size=1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))
        else:
            self.scale_gates = None
            self.gated_fuse = None

    def _branch_stat(self, feat: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(feat, dim=-1, keepdim=True)
        if self.fusion_use_maxpool:
            mx = torch.amax(feat, dim=-1, keepdim=True)
            return torch.cat([avg, mx], dim=1)
        return avg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        if self.fusion_mode == 'softgate_residual' and self.scale_gates is not None:
            base = self.fuse(torch.cat(feats, dim=1))
            temporal_att = None
            s_gate = self.cbam_softgate_strength
            if self.cbam_modulate_softgate and s_gate > 0.0 and (not isinstance(self.cbam, nn.Identity)) and hasattr(self.cbam, 'last_temporal_att'):
                _ = self.cbam(base)
                temporal_att = getattr(self.cbam, 'last_temporal_att', None)
            gated_feats = []
            for feat, gate_net in zip(feats, self.scale_gates):
                stat = self._branch_stat(feat)
                gate = torch.sigmoid(gate_net(stat))
                if temporal_att is not None and s_gate > 0.0:
                    gate_mult = 1.0 - s_gate + s_gate * temporal_att
                    gate = gate * gate_mult
                gated_feats.append(feat * gate)
            enh = self.gated_fuse(torch.cat(gated_feats, dim=1))
            out = base + self.fusion_residual_scale * enh
        else:
            out = self.fuse(torch.cat(feats, dim=1))
        out = self.cbam(out)
        return out

class PhaseNetUNet(nn.Module):

    def __init__(self, in_ch: int=3, n_class: int=3, depths: int=5, filters_root: int=8, kernels=(7,), pool_size: int=4, drop_rate: float=0.0, use_cbam: bool=False, use_separable: bool=False, fusion_mode: str='concat', fusion_gate_hidden: int=16, fusion_residual_scale: float=0.3, fusion_use_maxpool: bool=True, softgate_scope: str='all', use_temporal_bifpn_asff: bool=False, cbam_modulate_softgate: bool=True, cbam_softgate_strength: float=1.0) -> None:
        super().__init__()
        assert depths >= 1
        self.depths = depths
        self.pool_size = pool_size
        self.drop_rate = drop_rate
        self.kernel_size = kernels[0] if len(kernels) == 1 else 7
        self.kernels = kernels
        self.use_cbam = use_cbam
        self.use_separable = use_separable
        self.fusion_mode = fusion_mode
        self.fusion_gate_hidden = fusion_gate_hidden
        self.fusion_residual_scale = fusion_residual_scale
        self.fusion_use_maxpool = fusion_use_maxpool
        self.softgate_scope = softgate_scope
        self.use_temporal_bifpn_asff = use_temporal_bifpn_asff
        self.cbam_modulate_softgate = cbam_modulate_softgate
        self.cbam_softgate_strength = float(cbam_softgate_strength)

        def conv_block(in_c: int, out_c: int, fusion_mode_for_layer: str | None=None) -> nn.Module:
            local_fusion = fusion_mode_for_layer or self.fusion_mode
            if len(self.kernels) == 1 and self.kernels[0] == self.kernel_size and (not self.use_cbam) and (local_fusion == 'concat'):
                layers: list[nn.Module] = [nn.Conv1d(in_c, out_c, self.kernel_size, padding=self.kernel_size // 2, bias=False), nn.BatchNorm1d(out_c), nn.ReLU(inplace=True)]
                if drop_rate and drop_rate > 0.0:
                    layers.append(nn.Dropout(p=drop_rate))
                return nn.Sequential(*layers)
            return MultiScaleConv1d(in_ch=in_c, out_ch=out_c, kernels=self.kernels, dropout=drop_rate, use_separable=self.use_separable, use_cbam=self.use_cbam, fusion_mode=local_fusion, fusion_gate_hidden=self.fusion_gate_hidden, fusion_residual_scale=self.fusion_residual_scale, fusion_use_maxpool=self.fusion_use_maxpool, cbam_modulate_softgate=self.cbam_modulate_softgate, cbam_softgate_strength=self.cbam_softgate_strength)
        in_layers: list[nn.Module] = [nn.Conv1d(in_ch, filters_root, self.kernel_size, padding=self.kernel_size // 2, bias=False), nn.BatchNorm1d(filters_root), nn.ReLU(inplace=True)]
        if drop_rate and drop_rate > 0.0:
            in_layers.append(nn.Dropout(p=drop_rate))
        self.input_conv = nn.Sequential(*in_layers)
        downs: list[nn.Module] = []
        pools: list[nn.Module] = []
        ch_in = filters_root
        for d in range(depths):
            filters = int(2 ** d * filters_root)
            if self.fusion_mode == 'softgate_residual' and self.softgate_scope == 'deep':
                fm = 'softgate_residual' if d >= max(0, depths - 2) else 'concat'
            else:
                fm = self.fusion_mode
            downs.append(conv_block(ch_in, filters, fusion_mode_for_layer=fm))
            ch_in = filters
            if d < depths - 1:
                pool_layers: list[nn.Module] = [nn.Conv1d(ch_in, ch_in, kernel_size=self.kernel_size, stride=pool_size, padding=self.kernel_size // 2, bias=False), nn.BatchNorm1d(ch_in), nn.ReLU(inplace=True)]
                if drop_rate and drop_rate > 0.0:
                    pool_layers.append(nn.Dropout(p=drop_rate))
                pools.append(nn.Sequential(*pool_layers))
        self.downs = nn.ModuleList(downs)
        self.pools = nn.ModuleList(pools)
        if self.use_temporal_bifpn_asff:
            ch_list: list[int] = [int(2 ** d * filters_root) for d in range(depths)]
            self.ms_fusion = TemporalBiFPNASFF(ch_list)
        else:
            self.ms_fusion = None
        ups: list[nn.Module] = []
        up_convs: list[nn.Module] = []
        ch = ch_in
        for d in reversed(range(depths - 1)):
            filters = int(2 ** d * filters_root)
            up_convs.append(nn.ConvTranspose1d(ch, filters, kernel_size=pool_size, stride=pool_size, padding=0, bias=False))
            ch = filters * 2
            dec_idx = depths - 2 - d
            if self.fusion_mode == 'softgate_residual' and self.softgate_scope == 'deep':
                fm = 'softgate_residual' if dec_idx < 2 else 'concat'
            else:
                fm = self.fusion_mode
            ups.append(conv_block(ch, filters, fusion_mode_for_layer=fm))
            ch = filters
        self.up_convs = nn.ModuleList(up_convs)
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv1d(ch, n_class, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_temporal_att = None
        x = self.input_conv(x)
        downs: list[torch.Tensor] = []
        for i, block in enumerate(self.downs):
            x = block(x)
            downs.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        if self.ms_fusion is not None and len(downs) >= 5:
            p1, p2, p3, p4, p5 = (downs[0], downs[1], downs[2], downs[3], downs[4])
            f1, f2, f3, f4, f5 = self.ms_fusion(p1, p2, p3, p4, p5)
            downs[0] = f1
            downs[1] = f2
            downs[2] = f3
            downs[3] = f4
            downs[4] = f5
        for i, (up_conv, up_block) in enumerate(zip(self.up_convs, self.ups)):
            x = up_conv(x)
            skip = downs[-(i + 2)]
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                pad_left = diff // 2
                pad_right = diff - pad_left
                x = F.pad(x, (pad_left, pad_right))
            x = torch.cat([x, skip], dim=1)
            x = up_block(x)
        logits = self.head(x)
        last_att = None
        for m in self.modules():
            att = getattr(m, 'last_temporal_att', None)
            if att is not None and isinstance(att, torch.Tensor) and (att.numel() > 1):
                last_att = att
        self.last_temporal_att = last_att
        return logits
