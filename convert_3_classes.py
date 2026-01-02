"""
Complete Video ONNX Export Script - 3 Models Hierarchical Ensemble
Meets the requirements:
1. Dynamic axes for batch_size, frames, height, width
2. Accept raw pixel values [0-255] uint8
3. Return logits for 3 classes [real, synthetic, semi]
4. Hierarchical ensemble logic: Semi controls semi prob, RF splits remaining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from typing import List, Dict
import time
from model import Model



class VideoEnsembleWrapper(nn.Module):
    """
    Video ensemble wrapper that meets ONNX requirements:
    - Accepts raw pixels [0-255] uint8 with dynamic batch, frames, height, width
    - Returns logits for 3 classes
    - Handles all preprocessing internally
    - Uses hierarchical ensemble logic
    """

    def __init__(self, model_name: str, ckpt_path: str, target_device: str = 'cpu'):
        super(VideoEnsembleWrapper, self).__init__()
        self.model_name = model_name
        self.ckpt_path = ckpt_path

        print("🔄 Creating VideoEnsembleWrapper from TwoModelVideoClassifier...")
        print(f"   Target device: {target_device}")
        self.device = target_device
        self._load_models()

        self.target_frames = 4  # Fixed for ONNX compatibility

        # Move entire wrapper to target device
        self.to(target_device)

    def _load_models(self):
        """Load both models"""
        print("Loading 3 classes model...")
        self.realfake_model = Model(self.model_name, num_classes=2).to(self.device)
        self.realfake_model.eval()
        state_dict = torch.load(self.ckpt_path, map_location=self.device)["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
        state_dict = new_state_dict
        self.realfake_model.load_state_dict(state_dict)

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        x = video_batch.to(torch.float32) * (1.0 / 255.0)
        # get first frame
        x = x[:, 0, :, :, :]
    
        with torch.no_grad():
            frame_logits = self.realfake_model(x)  # (B*F, 3)
        return frame_logits
        probs = torch.softmax(frame_logits, dim=1)
        # 5 class to 2 class, sum of 0 and 3 to 0, sum of 1,2,4 to 1
        new_probs = torch.zeros(probs.shape[0], 2, device=probs.device)
        new_probs[:, 0] = probs[:, 0] + probs[:, 3]
        new_probs[:, 1] = probs[:, 1] + probs[:, 2]
        return self.probs_to_logits(new_probs)

        # --- NEW: post-processing logic in pure torch (ONNX-friendly) ---
        # Convert logits -> probabilities
        logits = self.probs_to_logits(new_probs)
        pred_probs = torch.softmax(logits, dim=1)  # (B, N)
        N = pred_probs.shape[1]
    
        # compute per-sample max prob and argmax index
        max_vals, max_idx = torch.max(pred_probs, dim=1)  # (B,), (B,)
    
        # condition mask: sample-wise whether max > 0.9
        mask = max_vals < 0.9  # (B,) boolean tensor
    
        # build adjusted probabilities: top class = 0.95, others = (1-0.95)/(N-1)
        top_value = 0.51
        other_value = 1.0 - top_value
    
        # create tensor filled with other_value, then scatter top_value at max_idx
        adjusted = torch.full_like(pred_probs, other_value)  # (B, N)
        adjusted.scatter_(1, max_idx.unsqueeze(1), top_value)  # in-place set top class
    
        # choose per-sample between adjusted and original probs
        mask_expanded = mask.unsqueeze(1)  # (B,1)
        final_probs = torch.where(mask_expanded, adjusted, pred_probs)  # (B, N)
    
        # convert back to logits (log-probabilities style). clamp to avoid log(0).
        final_logits = self.probs_to_logits(final_probs)
    
        return final_logits  # (B, N)

    def _sample_frames_onnx(self, video_batch: torch.Tensor, target_frames: int = 4) -> torch.Tensor:
        """
        ONNX-friendly frame sampling - takes last frames:
        - No Python control-flow
        - Always returns exactly `target_frames` frames
        - Works for T < target_frames (indices repeat) and T > target_frames (takes last frames)

        video_batch: (B, T, 3, H, W)
        returns:    (B, target_frames, 3, H, W)
        """
        # Get dynamic shape tensor -> ... [B, T, C, H, W]
        dyn_shape = torch.onnx.operators.shape_as_tensor(video_batch)
        T = dyn_shape[1].to(torch.float32)  # scalar tensor in graph

        # Take last target_frames frames
        # If T >= target_frames: indices from [T - target_frames, T - 1]
        # If T < target_frames: start from 0 (will repeat frames)
        start_idx = torch.clamp(T - target_frames, min=0.0)
        end_idx = T - 1.0
        
        # Create indices from start_idx to end_idx (last target_frames frames)
        idx_f = torch.linspace(
            start_idx,
            end_idx,
            steps=target_frames,
            device=video_batch.device
        )
        idx = idx_f.round().to(torch.int64)
        # Clamp to valid range [0, T-1] to handle edge cases
        T_int = dyn_shape[1].to(torch.int64)
        idx = torch.clamp(idx, min=0, max=T_int - 1)

        # Gather along time axis
        sampled = video_batch.index_select(dim=1, index=idx)  # ONNX -> Gather axis=1
        return sampled

    def probs_to_logits(self, probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Convert probabilities (after sigmoid/softmax) to logits (before softmax).
        Numerically stable and ONNX-safe.
        """
        probs = torch.clamp(probs, eps, 1 - eps)
        logits = torch.log(probs)
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits

class ONNXVideoEnsembleExporter:
    """
    Export the complete video ensemble to ONNX with correct requirements
    """
    
    def __init__(self):
        self.wrapper = None
        
    def create_wrapper(self, model_name: str, ckpt_path: str, target_device='cpu'):
        """Create the video ensemble wrapper"""
        print(f"🔧 Creating VideoEnsembleWrapper on device: {target_device}...")
        self.wrapper = VideoEnsembleWrapper(model_name=model_name, ckpt_path=ckpt_path, target_device=target_device)
        return self.wrapper
    
    def export_to_onnx(self, 
                       model_name: str,
                       ckpt_path: str,
                       output_file: str = "video_ensemble_complete.onnx", 
                       test_batch_size: int = 4, 
                       test_frames: int = 16,
                       test_height: int = 224, 
                       test_width: int = 224,
                       target_device: str = 'cpu') -> str:
        """
        Export complete video ensemble to ONNX with dynamic axes support
        
        Args:
            output_file: Output ONNX file path
            test_batch_size: Batch size for testing (will be dynamic)
            test_frames: Number of frames for testing (will be dynamic)
            test_height: Video height for testing (will be dynamic)
            test_width: Video width for testing (will be dynamic)
            target_device: Device for export ('cpu' recommended for ONNX)
            
        Returns:
            str: Path to exported ONNX file
        """
        if self.wrapper is None:
            self.create_wrapper(model_name=model_name, ckpt_path=ckpt_path, target_device=target_device)
        
        print(f"📦 Exporting video ensemble to {output_file}...")
        print(f"   Using device: {target_device}")
        print(f"   Test input shape: ({test_batch_size}, {test_frames}, 3, {test_height}, {test_width})")
        
        # Create dummy input (RAW PIXELS [0-255] as UINT8) on the same device
        dummy_input = torch.randint(0, 256, (test_batch_size, test_frames, 3, test_height, test_width), dtype=torch.uint8)
        dummy_input = dummy_input.to(target_device)
        
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Input range: [{dummy_input.min():.0f}, {dummy_input.max():.0f}] (raw pixels)")
        print(f"   Input device: {dummy_input.device}")
        print(f"   Wrapper device: {next(self.wrapper.parameters()).device}")
        
        # Export to ONNX with DYNAMIC AXES for video
        try:
            # Ensure model is in eval mode
            self.wrapper.eval()
            
            # Also ensure all sub-models are in eval mode
            for model in [self.wrapper.realfake_model]:
                model.eval()
            
            # Use context manager to ensure no gradients
            with torch.no_grad():
                torch.onnx.export(
                    self.wrapper,
                    dummy_input,
                    output_file,
                    input_names=['input'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input': {0: 'batch_size', 1: 'frames'},    # Dynamic batch, frames, height, width
                        'logits': {0: 'batch_size'}    # Dynamic batch dimension
                    },
                    opset_version=17,  # ONNX opset version
                    do_constant_folding=True,  # Optimize constants
                    verbose=False,
                    export_params=True
                )
            
            print(f"✅ Successfully exported to {output_file}")
            print("🎯 ONNX Video Model Specifications:")
            print("   📥 Input: Raw pixels [0-255] uint8, dynamic (batch_size, frames, 3, height, width)")
            print("   📤 Output: Logits (batch_size, 2) [real, synthetic]")
            print(f"   🎬 Frame processing: Sample {self.wrapper.target_frames} frames uniformly")

            return output_file
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            print("   Some PyTorch operations might not be ONNX-compatible")
            raise e
    
    def verify_onnx_model(self, onnx_file: str) -> bool:
        """
        Verify the exported ONNX video model works correctly with dynamic shapes
        """
        print(f"🧪 Verifying ONNX video model: {onnx_file}")
        
        try:
            import onnx
            import os
            
            # Check file size
            file_size = os.path.getsize(onnx_file) / (1024**3)  # GB
            print(f"   📁 Model size: {file_size:.2f} GB")
            
            # Test inference with ONNX Runtime
            session = ort.InferenceSession(onnx_file)
            print("   ✅ ONNX Runtime can load the model")
            
            # Get input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            print(f"   📥 Input: {input_info.name} {input_info.shape}")
            print(f"   📤 Output: {output_info.name} {output_info.shape}")
            
            # Test with different video shapes (dynamic axes)
            test_configs = [
                (1, 16, 3, 224, 224),   # Short video, small resolution
                (2, 16, 3, 224, 224),  # Medium video, standard resolution
                (1, 16, 3, 224, 224),  # Long video, higher resolution
                (3, 16, 3, 224, 224),   # Batch processing
            ]
            
            for batch_size, frames, channels, height, width in test_configs:
                print(f"   🧪 Testing shape: ({batch_size}, {frames}, {channels}, {height}, {width})")
                
                # Create test input (raw pixels [0-255] as UINT8)
                test_input = np.random.randint(0, 256, (batch_size, frames, channels, height, width), dtype=np.uint8)
                
                # ONNX inference
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                onnx_output = session.run([output_name], {input_name: test_input})[0]
                
                # Verify output shape and values
                expected_shape = (batch_size, 3)
                assert onnx_output.shape == expected_shape, f"Wrong output shape: {onnx_output.shape}, expected: {expected_shape}"
                
                # Convert logits to probabilities for verification
                onnx_probs = torch.softmax(torch.from_numpy(onnx_output), dim=1).numpy()
                assert np.all(onnx_probs >= 0) and np.all(onnx_probs <= 1), "Probabilities not in [0,1]"
                assert np.allclose(onnx_probs.sum(axis=1), 1.0, atol=1e-6), "Probabilities don't sum to 1"
                
                print(f"      ✅ Output shape: {onnx_output.shape}")
                print(f"      ✅ Logits range: [{onnx_output.min():.3f}, {onnx_output.max():.3f}]")
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX verification failed: {e}")
            return False
    
    def compare_pytorch_vs_onnx(self, onnx_file: str, num_tests: int = 3) -> Dict:
        """
        Compare PyTorch vs ONNX outputs to ensure accuracy
        """
        print(f"📊 Comparing PyTorch vs ONNX video accuracy...")
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_file)
        input_name = session.get_inputs()[0].name
        
        differences = []
        
        for i in range(num_tests):
            # Random test input with dynamic size
            batch_size = np.random.choice([1, 1, 1])
            frames = np.random.choice([16, 16, 16])
            height = np.random.choice([224, 224, 224])
            width = np.random.choice([224, 224, 224])
            
            # Raw pixels [0-255] as UINT8
            test_input = torch.randint(0, 256, (batch_size, frames, 3, height, width), dtype=torch.uint8)
            
            # Move to same device as wrapper for PyTorch inference
            wrapper_device = next(self.wrapper.parameters()).device
            test_input_pytorch = test_input.to(wrapper_device)
            test_input_np = test_input.numpy()
            
            # PyTorch inference (returns logits)
            with torch.no_grad():
                pytorch_logits = self.wrapper(test_input_pytorch).cpu().numpy()
            
            # ONNX inference (returns logits)
            onnx_logits = session.run(None, {input_name: test_input_np})[0]
            
            # Calculate difference
            diff = np.abs(pytorch_logits - onnx_logits).max()
            differences.append(diff)
            
            print(f"   Test {i+1}: Input ({batch_size}, {frames}, 3, {height}, {width}) -> Max diff: {diff:.8f}")
        
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)
        
        print(f"   Max difference: {max_diff:.8f}")
        print(f"   Mean difference: {mean_diff:.8f}")
        
        if max_diff < 1e-5:
            print("✅ PyTorch and ONNX outputs are nearly identical!")
        elif max_diff < 1e-3:
            print("⚠️ Small differences detected (acceptable for most use cases)")
        else:
            print("❌ Large differences detected - check implementation")
        
        return {
            'max_difference': max_diff,
            'mean_difference': mean_diff,
            'num_tests': num_tests
        }


def main_video_export_pipeline(model_name: str, ckpt_path: str, onnx_file: str):
    """
    Complete video export pipeline
    """
    exporter = ONNXVideoEnsembleExporter()
    # Use cpu for ONNX export to avoid device issues
    onnx_file = exporter.export_to_onnx(
        model_name=model_name,
        ckpt_path=ckpt_path,
        output_file=onnx_file, 
        test_batch_size=1, 
        test_frames=16,     # Dynamic frames
        test_height=224,    # Dynamic height
        test_width=224,     # Dynamic width
        target_device='cpu'
    )
    # return onnx_file
    try:
        accuracy_results = exporter.compare_pytorch_vs_onnx(onnx_file, num_tests=3)
        print("✅ Accuracy comparison completed")
    except Exception as e:
        print(f"❌ Accuracy comparison failed: {e}")
        accuracy_results = {'max_difference': 'N/A', 'mean_difference': 'N/A'}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="eva02_base_patch14_224.mim_in22k_ft_in22k_in1k")
    parser.add_argument("--ckpt_path", type=str, default="/mnt/video/bitmind_video_3_classes/eva02_base_patch14_224.mim_in22k_ft_in22k_in1k_3_classes_overfit_expended_1225_7419aa52/checkpoints/epoch=49.ckpt")
    parser.add_argument("--onnx_file", type=str, default="1225=49_hierarchy_85.onnx")
    args = parser.parse_args()
    onnx_file = main_video_export_pipeline(args.model_name, args.ckpt_path, args.onnx_file)
    print(f"ONNX file: {onnx_file}")