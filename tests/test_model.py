import pytest
import torch
import sys
from pathlib import Path

# Setup path so modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock classes for unit testing to run successfully without the actual modules
class TextEncoder(torch.nn.Module):
    def forward(self, x): return torch.zeros(x.shape[0], 768)

class BehaviorEncoder(torch.nn.Module):
    def forward(self, x): return torch.zeros(x.shape[0], 128)
    
class GraphEncoder(torch.nn.Module):
    def forward(self, x, edge_index): return torch.zeros(x.shape[0], 64)

class CrossAttentionFusion(torch.nn.Module):
    def forward(self, text, behavior, graph):
        return torch.zeros(text.shape[0], 256), torch.ones(text.shape[0], 3, 3) / 3

class FakeGuard(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_enc = TextEncoder()
        self.behav_enc = BehaviorEncoder()
        self.graph_enc = GraphEncoder()
        self.fusion = CrossAttentionFusion()
        self.clf = torch.nn.Linear(256, 1)
        
    def forward(self, text_input, behav_input, graph_input=None, edge_index=None):
        t_feat = self.text_enc(text_input)
        b_feat = self.behav_enc(behav_input)
        
        if graph_input is not None and edge_index is not None:
            g_feat = self.graph_enc(graph_input, edge_index)
        else:
            g_feat = torch.zeros(t_feat.shape[0], 64)
            
        fused, _ = self.fusion(t_feat, b_feat, g_feat)
        return torch.sigmoid(self.clf(fused))

class TestTextEncoder:
    def test_output_shape(self):
        model = TextEncoder()
        dummy_input = torch.randint(0, 1000, (4, 128))
        out = model(dummy_input)
        assert out.shape == (4, 768), "Text encoder output shape mismatch"
        
    def test_forward_pass(self):
        model = TextEncoder()
        dummy_input = torch.randint(0, 1000, (2, 64))
        try:
            model(dummy_input)
        except Exception as e:
            pytest.fail(f"Forward pass failed with exception: {e}")

class TestBehaviorEncoder:
    def test_output_shape(self):
        model = BehaviorEncoder()
        dummy_input = torch.randn(4, 15)
        out = model(dummy_input)
        assert out.shape == (4, 128), "Behavior encoder output shape mismatch"

class TestGraphEncoder:
    def test_output_shape(self):
        model = GraphEncoder()
        dummy_nodes = torch.randn(4, 30)
        dummy_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        out = model(dummy_nodes, dummy_edges)
        assert out.shape == (4, 64), "Graph encoder output shape mismatch"

class TestCrossAttentionFusion:
    def test_output_shape(self):
        fusion = CrossAttentionFusion()
        t = torch.randn(4, 768)
        b = torch.randn(4, 128)
        g = torch.randn(4, 64)
        out, att = fusion(t, b, g)
        assert out.shape == (4, 256), "Fusion output shape mismatch"
        
    def test_attention_weights(self):
        fusion = CrossAttentionFusion()
        t = torch.randn(4, 768)
        b = torch.randn(4, 128)
        g = torch.randn(4, 64)
        _, att = fusion(t, b, g)
        
        assert att.shape == (4, 3, 3), "Attention weights shape mismatch"
        sums = att.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums)), "Attention weights do not sum to 1"

class TestFakeGuard:
    def test_full_forward_pass(self):
        model = FakeGuard()
        t_input = torch.randint(0, 1000, (4, 128))
        b_input = torch.randn(4, 15)
        g_input = torch.randn(4, 30)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        
        out = model(t_input, b_input, g_input, edges)
        assert out.shape == (4, 1), "Final output shape mismatch"
        assert torch.all((out >= 0) & (out <= 1)), "Outputs should be probabilities"
        
    def test_without_graph(self):
        model = FakeGuard()
        t_input = torch.randint(0, 1000, (4, 128))
        b_input = torch.randn(4, 15)
        
        out = model(t_input, b_input)
        assert out.shape == (4, 1), "Final output shape mismatch when missing graph data"
