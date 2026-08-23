"""
Model architectures and neural network modules for FakeGuard.

Provides:
- TextEncoder: Transformer-based (BERT/RoBERTa) semantic representation
- BehaviorEncoder: Deep MLP encoder for user and product behavioral metadata
- GraphEncoder: PyTorch Geometric GNN (GAT/GCN/GraphSAGE) for reviewer-product relation graphs
- MultiViewFusion: Multi-head cross-attention fusion with attention weight explainability
- FakeGuardModel: End-to-end multi-view network combining all views
"""
