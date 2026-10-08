"""Aedge architecture used by the original fusion workflow (not Aedgethree_v2)."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(out_channels // 2, 8)
        self.skip = nn.Conv2d(skip_channels, hidden, 1)
        self.gate = nn.Conv2d(gate_channels, hidden, 1)
        self.score = nn.Sequential(nn.SiLU(inplace=True), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return skip * self.score(self.skip(skip) + self.gate(gate))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention = AttentionGate(skip_channels, in_channels, out_channels)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.attention(skip, x)], dim=1))


class EdgeGuidance(nn.Module):
    def __init__(self, channels: int, feature_channels: tuple[int, ...]) -> None:
        super().__init__()
        self.body = ConvBlock(3, channels)
        self.attention = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.edge_head = nn.Conv2d(channels, 1, 1)
        self.projections = nn.ModuleList(nn.Conv2d(channels, c, 1) for c in feature_channels)

    def forward(self, image: torch.Tensor, sizes: list[tuple[int, int]]) -> dict[str, object]:
        base = self.body(image.float())
        base = base * self.attention(base)
        features = [F.interpolate(base, size=size, mode="bilinear", align_corners=False) for size in sizes]
        return {"features": [p(f) for p, f in zip(self.projections, features)], "edge": self.edge_head(base)}


class EdgeFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.refine = ConvBlock(channels, channels)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        edge = self.proj(edge)
        return self.refine(x + edge + x * self.gate(edge))


class AdaptivePoreTemplate(nn.Module):
    def __init__(self, channels: int, num_templates: int = 4) -> None:
        super().__init__()
        self.templates = nn.Parameter(torch.randn(num_templates, channels))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        similarity = torch.einsum(
            "bchw,kc->bkhw", F.normalize(x, dim=1), F.normalize(self.templates, dim=1)
        )
        similarity = torch.sigmoid(similarity.max(dim=1, keepdim=True).values * self.scale.clamp(min=0.1))
        return x * (1 + similarity) + self.refine(torch.cat([x, similarity], dim=1)), similarity


class ROIZoomRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = ConvBlock(channels + 2, channels)
        self.delta = nn.Conv2d(channels, 1, 1)

    def forward(self, high_res: torch.Tensor, coarse: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
        roi = torch.sigmoid(coarse)
        template = F.interpolate(template, size=high_res.shape[-2:], mode="bilinear", align_corners=False)
        local = high_res * (1.0 + roi) + high_res * F.avg_pool2d(roi, 3, 1, 1)
        return self.delta(self.local(torch.cat([local, roi, template], dim=1)))


class AedgeNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_templates: int = 4) -> None:
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.enc1, self.enc2 = ConvBlock(in_channels, c), ConvBlock(c, c * 2)
        self.enc3, self.enc4 = ConvBlock(c * 2, c * 4), ConvBlock(c * 4, c * 8)
        self.bridge = ConvBlock(c * 8, c * 16)
        self.template = AdaptivePoreTemplate(c * 16, num_templates)
        self.roi_head = nn.Conv2d(c * 16, 1, 1)
        self.up4, self.up3 = UpBlock(c * 16, c * 8, c * 8), UpBlock(c * 8, c * 4, c * 4)
        self.up2, self.up1 = UpBlock(c * 4, c * 2, c * 2), UpBlock(c * 2, c, c)
        feature_channels = (c, c * 2, c * 4, c * 8)
        self.edge = EdgeGuidance(c, feature_channels)
        self.edge_fusions = nn.ModuleList([EdgeFusion(k) for k in feature_channels])
        self.mask_head = nn.Conv2d(c, 1, 1)
        self.edge_head = nn.Sequential(ConvBlock(c, c), nn.Conv2d(c, 1, 1))
        self.refine = ROIZoomRefinement(c)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bridge(self.pool(e4))
        b, template = self.template(b)
        roi = F.interpolate(self.roi_head(b), size=x.shape[-2:], mode="bilinear", align_corners=False)
        edge = self.edge(x, [e1.shape[-2:], e2.shape[-2:], e3.shape[-2:], e4.shape[-2:]])
        e1 = self.edge_fusions[0](e1, edge["features"][0])
        e2 = self.edge_fusions[1](e2, edge["features"][1])
        e3 = self.edge_fusions[2](e3, edge["features"][2])
        e4 = self.edge_fusions[3](e4, edge["features"][3])
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        coarse = self.mask_head(d1)
        refined = coarse + self.refine(
            d1,
            coarse,
            F.interpolate(template, size=d1.shape[-2:], mode="bilinear", align_corners=False),
        )
        return {
            "mask": refined,
            "coarse": coarse,
            "roi": roi,
            "edge": self.edge_head(d1),
            "edge_guidance": edge["edge"],
            "template": F.interpolate(template, size=x.shape[-2:], mode="bilinear", align_corners=False),
        }


SmallPoreSegNet = AedgeNet
