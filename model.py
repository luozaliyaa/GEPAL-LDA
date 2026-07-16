import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from torch_geometric.nn import GATConv, SAGEConv, GCNConv
from transformers import LlamaForCausalLM
from load_lnc_drug_data import build_gat_data


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels=in_dim, out_channels=hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(in_channels=hidden_dim * heads, out_channels=out_dim, heads=1, concat=True, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x  # [num_nodes, out_dim]


class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, dropout=0.3):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        return x  # [num_nodes, out_dim]


class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, dropout=0.3):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # [num_nodes, out_dim]


class Node2Prefix(nn.Module):
    def __init__(self, input_dim: int, llm_dim: int, num_prefix: int = 1):
        super().__init__()
        self.num_prefix = num_prefix
        self.adapter = nn.Linear(input_dim, llm_dim * num_prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        prefix = self.adapter(x)  # [batch, llm_dim * num_prefix]
        prefix = prefix.view(batch_size, self.num_prefix, -1)  # [batch, num_prefix, llm_dim]
        return prefix.to(x.dtype)


class MultiScaleAdapter(nn.Module):
    def __init__(self, input_dim: int, llm_dim: int, num_prefix: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.num_prefix = num_prefix

        # 分支1：线性投影 (粗粒度)
        self.proj_linear = nn.Linear(input_dim, llm_dim * num_prefix)

        # 分支2：MLP 投影 (细粒度/非线性)
        self.proj_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_dim * num_prefix)
        )

        # 可学习的融合权重 (也可以直接做平均)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        out_linear = self.proj_linear(x)  # [batch, llm_dim * num_prefix]
        out_mlp = self.proj_mlp(x)  # [batch, llm_dim * num_prefix]

        # 融合（可学习加权）
        fused = self.alpha * out_linear + (1 - self.alpha) * out_mlp

        prefix = fused.view(batch_size, self.num_prefix, -1)  # [batch, num_prefix, llm_dim]
        return prefix.to(x.dtype)


class EnhancedPromptAdapter(nn.Module):


    def __init__(self, input_dim: int, llm_dim: int, num_prefix: int = 1,
                 hidden_r: int = 64, slot_dim: int = 32, init_alpha: float = 0.5):
        super().__init__()
        self.num_prefix = num_prefix
        self.llm_dim = llm_dim

        # --- 低秩路径（控制参数量） input_dim -> r -> (llm_dim * num_prefix)
        self.fc1 = nn.Linear(input_dim, hidden_r)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_r, llm_dim * num_prefix)

        # --- 兼容旧 Node2Prefix：保留 adapter 线性层 (可选加载旧 checkpoint)
        # old Node2Prefix: self.adapter = nn.Linear(input_dim, llm_dim * num_prefix)
        self.adapter = nn.Linear(input_dim, llm_dim * num_prefix)

        # --- per-prefix slot embedding（每个 prefix 一个小向量，映射到 llm_dim）
        self.slot_emb = nn.Parameter(torch.randn(num_prefix, slot_dim) * 0.02)
        self.slot_proj = nn.Linear(slot_dim, llm_dim)

        # LayerNorm 与融合系数
        self.ln = nn.LayerNorm(llm_dim)
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))  # 融合 lowrank 与 direct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_dim]
        return: [batch, num_prefix, llm_dim]
        """
        bsz = x.size(0)

        # low-rank path
        low = self.fc2(self.act(self.fc1(x)))  # [bsz, llm_dim * num_prefix]
        low = low.view(bsz, self.num_prefix, self.llm_dim)  # [bsz, num_prefix, llm_dim]

        # direct path (兼容 old Node2Prefix)
        direct = self.adapter(x)  # [bsz, llm_dim * num_prefix]
        direct = direct.view(bsz, self.num_prefix, self.llm_dim)

        # slot path
        slot = self.slot_proj(self.slot_emb)  # [num_prefix, llm_dim]
        slot = slot.unsqueeze(0).expand(bsz, -1, -1)  # [bsz, num_prefix, llm_dim]

        # 融合： 默认 fused = alpha * low + (1-alpha) * direct，然后再和 slot 做简单融合（这里取加和）
        fused_main = self.alpha * low + (1.0 - self.alpha) * direct
        fused = fused_main + slot  # slot 起到固定角色补充信息的作用

        # LayerNorm
        fused = self.ln(fused)

        return fused.to(x.dtype)


class GATWithAdapterForLLM(nn.Module):
    def __init__(self,
                 model,
                 in_dim: int = 256,
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 heads: int = 4,
                 dropout: float = 0.3,
                 llm_dim: int = 4096,
                 num_prefix: int = 6,
                 ckpt_path: Optional[str] = None,
                 freeze_gnn: bool = False,
                 fold: int = 1,
                 ):
        super().__init__()
        self.llm = model
        data_obj = build_gat_data(fold=fold)
        self.node_features = data_obj.x
        self.edge_index = data_obj.edge_index
        print("node_features:", self.node_features)
        print("edge_index:", self.edge_index)

        # 1. 定义 GAT
        self.gat = GATModel(in_dim=in_dim,
                            hidden_dim=hidden_dim,
                            out_dim=out_dim,
                            heads=heads,
                            dropout=dropout)

        self.gcn = GCNModel(in_dim=in_dim,
                            hidden_dim=hidden_dim,
                            out_dim=out_dim,
                            dropout=dropout)

        self.sage = GraphSAGEModel(in_dim=in_dim,
                                   hidden_dim=hidden_dim,
                                   out_dim=out_dim,
                                   dropout=dropout)
        # gated 融合的 α
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # 2. 定义 Node2Prefix
        # self.node2prefix = Node2Prefix(input_dim=out_dim * 2,   # 因为拼接 lnc 和 drug
        #                                llm_dim=llm_dim,
        #                                num_prefix=num_prefix)

        self.node2prefix = MultiScaleAdapter(input_dim=out_dim * 2,
                                             llm_dim=llm_dim,
                                             num_prefix=num_prefix,
                                             hidden_dim=512)  # hidden_dim可调

        # 每个实体生成一半 prefix
        self.lnc_adapter = MultiScaleAdapter(
            input_dim=out_dim,
            llm_dim=llm_dim,
            num_prefix=num_prefix // 2,
            hidden_dim=512
        )

        self.drug_adapter = MultiScaleAdapter(
            input_dim=out_dim,
            llm_dim=llm_dim,
            num_prefix=num_prefix // 2,
            hidden_dim=512
        )

        self.diff_adapter = MultiScaleAdapter(
            input_dim=out_dim,
            num_prefix=num_prefix // 2,  # 你希望 strategy3 产生多少 prefix
            llm_dim=llm_dim,
            hidden_dim=512
        )


        # self.node2prefix = EnhancedPromptAdapter(
        #     input_dim=out_dim * 2,  # 因为拼接 lnc 和 drug
        #     llm_dim=llm_dim,
        #     num_prefix=num_prefix,
        #     hidden_r=64,  # bottleneck-r，可按显存调小或加大
        #     slot_dim=32,
        #     init_alpha=0.5
        # )

        # 3. 如果给定 ckpt_path，加载预训练参数
        if ckpt_path is not None:
            print(f"Loading GAT+GCN+Node2Prefix from {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(state, strict=False)

        if freeze_gnn:
            for p in self.gat.parameters():
                p.requires_grad = False
            for p in self.gcn.parameters():
                p.requires_grad = False
            for p in self.node2prefix.parameters():
                p.requires_grad = False
            print("GAT + GCN + Node2Prefix are frozen")

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                embedding_ids: torch.LongTensor = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):

        # 统一 device（非常重要）
        device = input_ids.device
        if self.node_features.device != device:
            self.node_features = self.node_features.to(device)
        if self.edge_index.device != device:
            self.edge_index = self.edge_index.to(device)

        node_features = self.node_features.to(device)
        edge_index = self.edge_index.to(device)

        # 1) 通过 GAT 得到节点表示
        gat_out = self.gat(node_features, edge_index)
        gcn_out = self.gcn(node_features, edge_index)
        sage_out = self.sage(node_features, edge_index)

        # --- gated 融合 ---
        # alpha = torch.sigmoid(self.alpha)
        # lnc_feats = alpha * gat_out[embedding_ids[:, 0]] + (1 - alpha) * gcn_out[embedding_ids[:, 0]]
        # drug_feats = alpha * gat_out[embedding_ids[:, 1]] + (1 - alpha) * gcn_out[embedding_ids[:, 1]]
        fused_out = torch.cat([gat_out, gcn_out], dim=-1)
        fused_out = self.proj(fused_out)

        lnc_feats = fused_out[embedding_ids[:, 0]]  # [batch, out_dim]
        drug_feats = fused_out[embedding_ids[:, 1]]  # [batch, out_dim]

        # strategy1
        pair_feats = torch.cat([lnc_feats, drug_feats], dim=-1)
        prefix_embeds1 = self.node2prefix(pair_feats)  # [batch, num_prefix, llm_dim]

        # strategy2
        # 各自生成 prefix
        prefix_lnc = self.lnc_adapter(lnc_feats)  # [batch, num_prefix/2, llm_dim]
        prefix_drug = self.drug_adapter(drug_feats)  # [batch, num_prefix/2, llm_dim]
        # 拼接 prefix
        prefix_embeds2 = torch.cat([prefix_lnc, prefix_drug], dim=1)

        # strategy3
        diff_feats = torch.abs(lnc_feats - drug_feats)  # [batch, out_dim]
        prefix_embeds3 = self.diff_adapter(diff_feats)

        # # 将特征置为0
        # prefix_embeds = torch.zeros_like(prefix_embeds)
        # prefix_embeds = torch.cat([prefix_embeds2, prefix_embeds3], dim=1)
        # prefix_embeds = torch.cat(
        #     [prefix_embeds1, prefix_embeds2, prefix_embeds3],
        #     dim=1
        # )
        prefix_embeds = prefix_embeds1

        token_embeds = self.llm.model.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        batch_size, prefix_len, _ = prefix_embeds.shape
        prefix_mask = torch.ones((batch_size, prefix_len), device=attention_mask.device)
        new_attention_mask = torch.cat([prefix_mask, attention_mask], dim=-1)

        prefix_labels = torch.full((batch_size, prefix_len), -100, dtype=torch.long, device=labels.device)
        new_labels = torch.cat([prefix_labels, labels], dim=-1)

        return self.llm(
            input_ids=None,
            attention_mask=new_attention_mask,
            inputs_embeds=inputs_embeds,
            labels=new_labels,
            return_dict=return_dict,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

