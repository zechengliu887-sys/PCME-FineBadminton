from typing import Dict, List, Union

import torch
import torch.nn as nn

from mmaction.registry import MODELS
from mmaction.models.heads.base import BaseHead


@MODELS.register_module()
class PairOnlyTop2CalibrationHead(BaseHead):
    """Parameter-matched pair-only top-two calibration control.

    This control intentionally preserves:
      - the original GCN classification path;
      - detached ordered Top-2 candidate routing;
      - class embeddings;
      - directional ordered pair encoding;
      - a shared directional scorer;
      - antisymmetric +r / -r logit correction.

    It intentionally removes:
      - feature-key projection;
      - temporal-joint attention;
      - spatiotemporal evidence aggregation.

    The calibration branch therefore has no access to the
    intermediate T x V motion feature after preliminary logits
    have been computed.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        pair_dim: int = 64,
        hidden_dim: int = 716,
        loss_cls: Dict = dict(
            type="CrossEntropyLoss"
        ),
        dropout: float = 0.0,
        average_clips: str = "prob",
        init_cfg: Union[
            Dict,
            List[Dict]
        ] = dict(
            type="Normal",
            layer="Linear",
            std=0.01,
        ),
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            average_clips=average_clips,
            init_cfg=init_cfg,
            **kwargs,
        )

        self.pair_dim = int(pair_dim)
        self.hidden_dim = int(hidden_dim)

        # Original GCNHead-compatible classification path.
        self.pool = nn.AdaptiveAvgPool2d(1)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.fc = nn.Linear(
            in_channels,
            num_classes,
        )

        # Same ordered pair representation as Full PCME.
        self.class_embedding = nn.Embedding(
            num_classes,
            pair_dim,
        )

        self.pair_encoder = nn.Sequential(
            nn.Linear(
                pair_dim * 3,
                pair_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                pair_dim,
                pair_dim,
            ),
        )

        # Pair-only calibration scorer.
        #
        # Input:
        #   D-dimensional ordered pair query
        #   + 3 detached preliminary-logit descriptors.
        #
        # Hidden width 716 is fixed only by parameter matching
        # to Full PCME, before any validation experiment.
        self.calibration_scorer = nn.Sequential(
            nn.Linear(
                pair_dim + 3,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                1,
                bias=False,
            ),
        )

        # Exact zero-correction initialization, matching
        # Full PCME's initialization principle.
        nn.init.zeros_(
            self.calibration_scorer[-1].weight
        )

    def init_weights(self):
        super().init_weights()

        nn.init.zeros_(
            self.calibration_scorer[-1].weight
        )

    def _pair_query(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:

        e1 = self.class_embedding(first)
        e2 = self.class_embedding(second)

        pair_input = torch.cat(
            [
                e1,
                e2,
                e1 - e2,
            ],
            dim=1,
        )

        return self.pair_encoder(
            pair_input
        )

    def _baseline_logits(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x: [N, M, C, T, V]
        N, M, C, T, V = x.shape

        if C != self.in_channels:
            raise RuntimeError(
                f"Expected C={self.in_channels}, got C={C}"
            )

        pooled = x.reshape(
            N * M,
            C,
            T,
            V,
        )

        pooled = self.pool(
            pooled
        )

        pooled = pooled.reshape(
            N,
            M,
            C,
        )

        pooled = pooled.mean(
            dim=1
        )

        if self.dropout is not None:
            pooled = self.dropout(
                pooled
            )

        return self.fc(
            pooled
        )

    def _calibration_score(
        self,
        query: torch.Tensor,
        first_logit: torch.Tensor,
        second_logit: torch.Tensor,
    ) -> torch.Tensor:

        # Descriptor is explicitly detached.
        descriptor = torch.stack(
            [
                first_logit.detach(),
                second_logit.detach(),
                (
                    first_logit
                    - second_logit
                ).detach(),
            ],
            dim=1,
        )

        calibration_input = torch.cat(
            [
                query,
                descriptor,
            ],
            dim=1,
        )

        return (
            self.calibration_scorer(
                calibration_input
            )
            .squeeze(1)
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        # =====================================================
        # 1. Preliminary classifier
        # =====================================================

        z0 = self._baseline_logits(
            x
        )

        # =====================================================
        # 2. Detached ordered Top-2 routing
        # =====================================================

        top2 = torch.topk(
            z0.detach(),
            k=2,
            dim=1,
        ).indices

        a = top2[:, 0]
        b = top2[:, 1]

        # =====================================================
        # 3. Ordered pair queries
        # =====================================================

        q_ab = self._pair_query(
            a,
            b,
        )

        q_ba = self._pair_query(
            b,
            a,
        )

        # =====================================================
        # 4. Pair-only logit descriptors
        #
        # IMPORTANT:
        # No T x V feature is accessed below this point.
        # =====================================================

        za = (
            z0.gather(
                1,
                a.unsqueeze(1),
            )
            .squeeze(1)
        )

        zb = (
            z0.gather(
                1,
                b.unsqueeze(1),
            )
            .squeeze(1)
        )

        s_ab = self._calibration_score(
            q_ab,
            za,
            zb,
        )

        s_ba = self._calibration_score(
            q_ba,
            zb,
            za,
        )

        correction = 0.5 * (
            s_ab
            - s_ba
        )

        # =====================================================
        # 5. Same antisymmetric local reranking as Full PCME
        # =====================================================

        delta = torch.zeros_like(
            z0
        )

        delta.scatter_add_(
            1,
            a.unsqueeze(1),
            correction.unsqueeze(1),
        )

        delta.scatter_add_(
            1,
            b.unsqueeze(1),
            -correction.unsqueeze(1),
        )

        return z0 + delta
