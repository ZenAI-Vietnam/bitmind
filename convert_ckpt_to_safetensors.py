import torch 
import os 
import argparse
from safetensors.torch import save_file

def convert_ckpt_to_safetensors(ckpt_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
        print("Created directory to save: ", output_path)
        
    try:
        # Load the checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Extract the state_dict from the checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
            
        rename_rules = {
            "vision_embeddings": "vision_model.embeddings",
            "vision_encoder": "vision_model.encoder",
            "vision_pre_layernorm": "vision_model.pre_layernorm",
            "vision_post_layernorm": "vision_model.post_layernorm"
        }
        
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("model."):
                new_key = new_key.replace("model.", "")
            
            for old_name, new_name in rename_rules.items():
                if old_name in new_key:
                    new_key = new_key.replace(old_name, new_name)
            
            new_state_dict[new_key] = v
        
        # Save the state_dict in safetensors format
        final_output = os.path.join(output_path, 'model.safetensors')
        save_file(new_state_dict, final_output)
        
        print("Conversion successful! Saved to: ", final_output)
    except Exception as e:
        print("Error during conversion: ", str(e))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to safetensors format")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the input checkpoint file (.ckpt or .pth)")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the converted safetensors file")
    
    args = parser.parse_args()
    
    convert_ckpt_to_safetensors(args.ckpt_path, args.output_path)