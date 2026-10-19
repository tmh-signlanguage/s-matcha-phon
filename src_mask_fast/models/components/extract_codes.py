import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

# src.models.components.
from src_mask_fast.models.components.train_multipart_vqvae import (
    ResidualVQVAE,
    load_dataset,
    separate_body_hands,
)


class CodeExtractor:
    """
    Extract discrete codes from motion sequences using trained VQ-VAE models
    """
    
    def __init__(self, checkpoint_dir: str, device: str = 'auto'):
        """
        Initialize code extractor
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            device: 'auto', 'cuda', or 'cpu'
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load models
        self.models = {}
        self.configs = {}
        self.input_dims = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load all three models"""
        print(f"\nLoading models from {self.checkpoint_dir}...")
        
        for part_name in ['body', 'r_hand', 'l_hand']:
            checkpoint_path = self.checkpoint_dir / f'{part_name}_model.pt'
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            config = checkpoint['config']
            input_dim = checkpoint['input_dim']
            
            # Initialize model
            model = ResidualVQVAE(
                input_dim=input_dim,
                hidden_dim=config['hidden_dim'],
                latent_dim=config['latent_dim'],
                num_quantizers=config['num_quantizers'],
                codebook_size=config['codebook_size'],
                commitment_weight=config['commitment_weight'],
            )
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            self.models[part_name] = model
            self.configs[part_name] = config
            self.input_dims[part_name] = input_dim
            
            print(f"  {part_name}: input_dim={input_dim}, "
                  f"num_quantizers={config['num_quantizers']}, "
                  f"codebook_size={config['codebook_size']}")
        
        print("Models loaded successfully!\n")
    
    def extract_codes_from_motion(
        self,
        motion: np.ndarray,
        return_latents: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract codes from a single motion sequence
        
        Args:
            motion: Full motion sequence [n_frames, n_features]
            return_latents: If True, also return continuous latent codes
        
        Returns:
            codes: Dictionary with codes per body part
                   Each part has shape [n_frames, num_quantizers]
                   Values are integers in range [0, codebook_size)
        """
        # Split motion into body parts
        body, r_hand, l_hand = separate_body_hands(motion)
        
        codes = {}
        if return_latents:
            latents = {}
        
        with torch.no_grad():
            for part_name, part_motion in [
                ('body', body),
                ('r_hand', r_hand),
                ('l_hand', l_hand)
            ]:
                # Convert to tensor [n_frames, n_features]
                part_tensor = torch.FloatTensor(part_motion).to(self.device)
                n_frames, n_features = part_tensor.shape
                
                # Get model
                model = self.models[part_name]
                
                # Encode to latent space
                z = model.encode(part_tensor)  # [n_frames, latent_dim]
                
                # Quantize to get codes
                z_q, indices, _ = model.quantize(z)

                if return_latents:
                    # print("z_q.shape: ", z_q.shape)  # [n_frames, latent_dim]
                    latents[part_name] = z_q.cpu().numpy()

                # print(f"Indices shape in quantizer {indices.shape}")  torch.Size([314, 4]) but before saving it is trasnposed to (4, n_frames)
                
                # indices is a list of tensors, one per quantizer
                # Each tensor has shape [n_frames]
                # Stack them to get [n_frames, num_quantizers]
                codes_array = np.stack([idx.cpu().numpy() for idx in indices], axis=1)
                # print(f"Codes array shape {codes_array.shape} int part {part_name}")
                codes[part_name] = codes_array.astype(np.int32)
        
        if return_latents:
            return codes, latents
        return codes
    
    def extract_codes_from_folder(
        self,
        input_folder: str,
        output_folder: str,
        save_latents: bool = False,
        save_reconstructions: bool = False
    ):
        """
        Extract codes from all motion files in a folder
        
        Args:
            input_folder: Folder containing .npz motion files
            output_folder: Folder to save extracted codes
            save_latents: If True, also save continuous latent codes
            save_reconstructions: If True, also save reconstructed motions
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Load all data
        print(f"Loading motion files from {input_folder}...")
        filenames, motions, _, _, _ = load_dataset(input_folder)
        print(f"Found {len(motions)} motion files\n")
        
        # Extract codes for each file
        print("Extracting codes...")
        
        all_stats = {
            'body': {'min': [], 'max': [], 'unique_per_frame': []},
            'r_hand': {'min': [], 'max': [], 'unique_per_frame': []},
            'l_hand': {'min': [], 'max': [], 'unique_per_frame': []},
        }
        
        for i, (filename, motion) in enumerate(tqdm(zip(filenames, motions), total=len(motions))):
            # Extract codes
            if save_latents:
                codes, latents = self.extract_codes_from_motion(motion, return_latents=True)
                # print(f"Latents body shape: {latents["body"].shape}")
            else:
                codes = self.extract_codes_from_motion(motion, return_latents=False)
            
            # Prepare data to save
            save_data = {
                'filename': filename,
                'n_frames': motion.shape[0],
                'original_shape': motion.shape,
            }
            
            # Add codes for each part
            for part_name in ['body', 'r_hand', 'l_hand']:
                part_codes = codes[part_name]
                save_data[f'{part_name}_codes'] = part_codes
                save_data[f'{part_name}_codes_shape'] = part_codes.shape
                
                # Collect statistics
                all_stats[part_name]['min'].append(part_codes.min())
                all_stats[part_name]['max'].append(part_codes.max())
                
                # Count unique codes per frame (across all quantizers)
                unique_per_frame = [len(np.unique(part_codes[frame])) for frame in range(part_codes.shape[0])]
                all_stats[part_name]['unique_per_frame'].extend(unique_per_frame)
            
            # Add latents if requested
            if save_latents:
                for part_name in ['body', 'r_hand', 'l_hand']:
                    save_data[f'{part_name}_latents'] = latents[part_name]
            
            # Add reconstructions if requested
            if save_reconstructions:
                body, r_hand, l_hand = separate_body_hands(motion)
                
                with torch.no_grad():
                    # Reconstruct each part
                    body_recon = self._reconstruct_part('body', body)
                    r_hand_recon = self._reconstruct_part('r_hand', r_hand)
                    l_hand_recon = self._reconstruct_part('l_hand', l_hand)
                
                save_data['body_recon'] = body_recon
                save_data['r_hand_recon'] = r_hand_recon
                save_data['l_hand_recon'] = l_hand_recon
                
                # Full reconstruction
                full_recon = np.concatenate([body_recon, r_hand_recon, l_hand_recon], axis=1)
                save_data['full_recon'] = full_recon
                save_data['recon_mse'] = np.mean((motion - full_recon) ** 2)
            
            # Save to file
            output_path = output_folder / f"{filename}_codes.npz"
            np.savez_compressed(output_path, **save_data)
        
        print(f"\nSaved codes to {output_folder}")
        
        # Print statistics
        self._print_statistics(all_stats)
        
        # Save metadata
        metadata = {
            'num_files': len(motions),
            'input_folder': str(input_folder),
            'output_folder': str(output_folder),
            'models': {
                part: {
                    'input_dim': self.input_dims[part],
                    'num_quantizers': self.configs[part]['num_quantizers'],
                    'codebook_size': self.configs[part]['codebook_size'],
                    'latent_dim': self.configs[part]['latent_dim'],
                }
                for part in ['body', 'r_hand', 'l_hand']
            },
            'statistics': {
                part: {
                    'code_min': int(np.min(stats['min'])),
                    'code_max': int(np.max(stats['max'])),
                    'avg_unique_per_frame': float(np.mean(stats['unique_per_frame'])),
                }
                for part, stats in all_stats.items()
            }
        }
        
        metadata_path = output_folder / 'codes_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_path}")
    
    def _reconstruct_part(self, part_name: str, part_motion: np.ndarray) -> np.ndarray:
        """Reconstruct a single body part"""
        part_tensor = torch.FloatTensor(part_motion).unsqueeze(0).to(self.device)
        model = self.models[part_name]
        recon_tensor, _, _ = model(part_tensor)
        return recon_tensor.squeeze(0).cpu().numpy()
    
    def _print_statistics(self, all_stats: Dict):
        """Print code statistics"""
        print("\n" + "="*60)
        print("CODE STATISTICS")
        print("="*60)
        
        for part_name, stats in all_stats.items():
            print(f"\n{part_name.upper()}:")
            print(f"  Code range: [{np.min(stats['min'])}, {np.max(stats['max'])}]")
            print(f"  Avg unique codes per frame: {np.mean(stats['unique_per_frame']):.2f}")
            print(f"  Max unique codes per frame: {np.max(stats['unique_per_frame'])}")
    
    def get_code_info(self) -> Dict:
        """
        Get information about code shapes and ranges
        
        Returns:
            Dictionary with code information for each part
        """
        info = {}
        for part_name in ['body', 'r_hand', 'l_hand']:
            config = self.configs[part_name]
            info[part_name] = {
                'num_quantizers': config['num_quantizers'],
                'codebook_size': config['codebook_size'],
                'code_shape_per_frame': [config['num_quantizers']],  # Each frame -> [num_quantizers]
                'code_range': [0, config['codebook_size'] - 1],
                'dtype': 'int32',
            }
        return info


def load_codes_from_file(filepath: str) -> Dict:
    """
    Load extracted codes from a saved .npz file
    
    Args:
        filepath: Path to saved codes file
    
    Returns:
        Dictionary containing codes and metadata
    """
    data = np.load(filepath)
    
    result = {
        'filename': str(data['filename']),
        'n_frames': int(data['n_frames']),
        'original_shape': tuple(data['original_shape']),
        'codes': {
            'body': data['body_codes'],
            'r_hand': data['r_hand_codes'],
            'l_hand': data['l_hand_codes'],
        },
        'code_shapes': {
            'body': tuple(data['body_codes_shape']),
            'r_hand': tuple(data['r_hand_codes_shape']),
            'l_hand': tuple(data['l_hand_codes_shape']),
        }
    }
    
    # Optional fields
    if 'body_latents' in data:
        result['latents'] = {
            'body': data['body_latents'],
            'r_hand': data['r_hand_latents'],
            'l_hand': data['l_hand_latents'],
        }
    
    if 'full_recon' in data:
        result['reconstruction'] = data['full_recon']
        result['recon_mse'] = float(data['recon_mse'])
    
    return result


def demonstrate_code_usage():
    """
    Demonstrate how to use extracted codes
    """
    print("\n" + "="*60)
    print("EXAMPLE: Using Extracted Codes")
    print("="*60)
    
    # Load a sample codes file
    codes_folder = Path('./codes_output')
    code_files = list(codes_folder.glob('*_codes.npz'))
    
    if not code_files:
        print("No code files found. Run extraction first.")
        return
    
    # Load first file
    sample_file = code_files[0]
    print(f"\nLoading: {sample_file.name}")
    
    codes_data = load_codes_from_file(sample_file)
    
    print(f"\nFile: {codes_data['filename']}")
    print(f"Frames: {codes_data['n_frames']}")
    print(f"Original motion shape: {codes_data['original_shape']}")
    
    print("\nCode shapes per body part:")
    for part, shape in codes_data['code_shapes'].items():
        print(f"  {part}: {shape}")
        print(f"    Per frame: [{shape[1]}] values")
        print(f"    Example frame 0: {codes_data['codes'][part][0]}")
    
    # Example: Using codes for sequence modeling
    print("\n" + "="*60)
    print("Example: Preparing codes for transformer input")
    print("="*60)
    
    # Concatenate all codes per frame
    body_codes = codes_data['codes']['body']  # [n_frames, num_quantizers]
    r_hand_codes = codes_data['codes']['r_hand']
    l_hand_codes = codes_data['codes']['l_hand']
    
    # Option 1: Concatenate all quantizers from all parts
    all_codes_per_frame = np.concatenate([body_codes, r_hand_codes, l_hand_codes], axis=1)
    print(f"\nOption 1 - Concatenate all codes:")
    print(f"  Shape: {all_codes_per_frame.shape}")
    print(f"  Each frame has {all_codes_per_frame.shape[1]} code indices")
    print(f"  Frame 0: {all_codes_per_frame[0]}")
    
    # Option 2: Use only first quantizer from each part (coarse codes)
    coarse_codes = np.stack([
        body_codes[:, 0],
        r_hand_codes[:, 0],
        l_hand_codes[:, 0]
    ], axis=1)
    print(f"\nOption 2 - Use only coarse codes (first quantizer):")
    print(f"  Shape: {coarse_codes.shape}")
    print(f"  Each frame has {coarse_codes.shape[1]} code indices (one per part)")
    print(f"  Frame 0: {coarse_codes[0]}")
    
    # Option 3: Flatten into a single sequence
    flat_codes = all_codes_per_frame.flatten()
    print(f"\nOption 3 - Flatten all codes:")
    print(f"  Shape: {flat_codes.shape}")
    print(f"  Total codes: {len(flat_codes)}")
    print(f"  First 12 codes: {flat_codes[:12]}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract VQ codes from motion data')
    parser.add_argument('--checkpoint_dir', type=str, default="/nfs/deepspeech/home/annkle/runs/1024_4/",  # "/nfs/deepspeech/home/annkle/runs" 512 by 256
                        help='Directory containing trained models')
    parser.add_argument('--input_folder', type=str, default="/nfs/deepspeech/home/annkle/lightning-hydra-template/data/data/5processed_mocap_body_and_fingers_val",
                        help='Folder containing motion .npz files')
    parser.add_argument('--output_folder', type=str, default="/nfs/deepflow/home/annkle/rvq-codes-for-expmaps-1024_4",
                        help='Folder to save extracted codes')
    parser.add_argument('--save_latents', action='store_true',
                        help='Save continuous latent codes in addition to discrete codes')
    parser.add_argument('--save_reconstructions', action='store_true',
                        help='Save reconstructed motions')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--demo', action='store_true',
                        help='Run demonstration of code usage')
    
    args = parser.parse_args()
    
    print(f"Do we save latents? {args.save_latents}")

    # Initialize extractor
    extractor = CodeExtractor(args.checkpoint_dir, device=args.device)
    
    # Print code information
    print("\n" + "="*60)
    print("CODE INFORMATION")
    print("="*60)
    code_info = extractor.get_code_info()
    for part_name, info in code_info.items():
        print(f"\n{part_name.upper()}:")
        print(f"  Number of quantizers: {info['num_quantizers']}")
        print(f"  Codebook size: {info['codebook_size']}")
        print(f"  Code shape per frame: {info['code_shape_per_frame']}")
        print(f"  Code value range: {info['code_range']}")
        print(f"  Data type: {info['dtype']}")
    
    # Extract codes
    extractor.extract_codes_from_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        save_latents=args.save_latents,
        save_reconstructions=args.save_reconstructions,
    )
    
    # Run demo if requested
    """if args.demo:
        demonstrate_code_usage()"""
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nExtracted codes saved to: {args.output_folder}")
    print("\nTo load codes in your own script:")
    print("  from extract_codes import load_codes_from_file")
    print("  codes_data = load_codes_from_file('path/to/file_codes.npz')")
    print("  body_codes = codes_data['codes']['body']  # Shape: [n_frames, num_quantizers]")


if __name__ == "__main__":
    main()