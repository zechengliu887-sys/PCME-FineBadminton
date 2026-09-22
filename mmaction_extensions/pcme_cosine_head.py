import math
from typing import Dict, List, Union

import torch
import torch.nn as nn

from mmaction.registry import MODELS
from mmaction.models.heads.base import BaseHead


@MODELS.register_module()
class CosineBidirectionalPairConditionedEvidenceHead(BaseHead):
    """Bidirectional pair-conditioned motion evidence head.

    Baseline:
        F -> GAP -> fc -> z0

    Candidate classes:
        a = Top-1(z0)
        b = Top-2(z0)

    Ordered evidence:
        s_ab = H(F, q(a,b))
        s_ba = H(F, q(b,a))

    Antisymmetric correction:
        r = 0.5 * (s_ab - s_ba)

        z[a] += r
        z[b] -= r

    No ground-truth label is used for candidate construction.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        pair_dim: int = 64,
        hidden_dim: int = 128,
        temperature: float = 8.0,
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

        self.pair_dim = pair_dim
        self.temperature = float(temperature)
        if self.temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        # --------------------------------------------------
        # Original GCNHead-compatible classification path.
        # --------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.fc = nn.Linear(
            in_channels,
            num_classes,
        )

        # --------------------------------------------------
        # Ordered pair query.
        #
        # q(a,b) != q(b,a)
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Convert each joint-time feature to key space.
        # --------------------------------------------------
        self.feature_key = nn.Conv2d(
            in_channels,
            pair_dim,
            kernel_size=1,
            bias=False,
        )

        # --------------------------------------------------
        # IMPORTANT:
        #
        # Scorer receives ONLY sample evidence.
        # q is NOT concatenated here.
        #
        # Therefore the pair query can determine
        # where to look, but cannot directly produce
        # a class-pair prior correction.
        #
        # Final layer has no bias.
        # --------------------------------------------------
        self.evidence_scorer = nn.Sequential(
            nn.Linear(
                in_channels,
                hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                1,
                bias=False,
            ),
        )

        # Zero correction at construction.
        nn.init.zeros_(
            self.evidence_scorer[-1].weight
        )

    def init_weights(self):
        # Standard MMEngine initialization first.
        super().init_weights()

        # Restore exact zero-correction starting point.
        nn.init.zeros_(
            self.evidence_scorer[-1].weight
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

    def _evidence_score(
        self,
        feat: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Extract query-conditioned evidence and score it.

        Args:
            feat:  [N, C, T, V]
            key:   [N, D, T, V]
            query: [N, D]

        Returns:
            score: [N]
        """

        N, C, T, V = feat.shape

        # --------------------------------------------------
        # PCME-V1.2:
        # scaled cosine pair-conditioned attention.
        #
        # V1.1 used:
        #     q @ k / sqrt(D)
        #
        # Validation diagnostics showed that this produced
        # almost uniform attention. Normalize q and k first
        # so the ordered-pair direction controls attention
        # independently of their raw vector magnitudes.
        # --------------------------------------------------

        query_norm = (
            query
            /
            (
                torch.linalg.vector_norm(
                    query,
                    dim=1,
                    keepdim=True,
                )
                + 1e-12
            )
        )

        key_norm = (
            key
            /
            (
                torch.linalg.vector_norm(
                    key,
                    dim=1,
                    keepdim=True,
                )
                + 1e-12
            )
        )

        attn_logits = (
            self.temperature
            * torch.einsum(
                "nd,ndtv->ntv",
                query_norm,
                key_norm,
            )
        )

        attn = torch.softmax(
            attn_logits.reshape(
                N,
                T * V,
            ),
            dim=1,
        )

        attn = attn.reshape(
            N,
            T,
            V,
        )

        evidence = torch.einsum(
            "ntv,nctv->nc",
            attn,
            feat,
        )

        return (
            self.evidence_scorer(
                evidence
            )
            .squeeze(1)
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:

        # x: [N, M, C, T, V]
        N, M, C, T, V = x.shape

        if C != self.in_channels:
            raise RuntimeError(
                f"Expected C={self.in_channels}, got C={C}"
            )

        # ==================================================
        # 1. Original baseline classifier
        # ==================================================

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

        z0 = self.fc(
            pooled
        )

        # ==================================================
        # 2. Candidate Top-2
        #
        # No GT labels.
        # ==================================================

        top2 = torch.topk(
            z0.detach(),
            k=2,
            dim=1,
        ).indices

        a = top2[:, 0]
        b = top2[:, 1]

        # ==================================================
        # 3. Ordered queries in BOTH directions
        # ==================================================

        q_ab = self._pair_query(
            a,
            b,
        )

        q_ba = self._pair_query(
            b,
            a,
        )

        # Target-player setting M=1.
        feat = x.mean(
            dim=1
        )
        # [N,C,T,V]

        key = self.feature_key(
            feat
        )

        # ==================================================
        # 4. Bidirectional sample evidence
        # ==================================================

        s_ab = self._evidence_score(
            feat,
            key,
            q_ab,
        )

        s_ba = self._evidence_score(
            feat,
            key,
            q_ba,
        )

        # Shared scorer bias/constant components cancel.
        correction = 0.5 * (
            s_ab
            - s_ba
        )

        # ==================================================
        # 5. Antisymmetric reranking
        # ==================================================

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
