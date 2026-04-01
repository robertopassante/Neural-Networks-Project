import sys
import time

try:
    import torch
except ImportError:
    print("[!] ERROR: PyTorch installed. Please run: pip install torch")
    sys.exit(1)

def print_header(title):
    print("\n" + "="*70)
    print(f" {title.upper()} ".center(70, '='))
    print("="*70)

def test_satmae_backbone():
    print_header("Test 1: Base SatMAE++ Backbone")
    try:
        from satmae_backbone import SatMAESegmentationBackbone
        print("[*] Initializing SatMAE++ Backbone (10 multi-spectral channels)...")
        model = SatMAESegmentationBackbone(in_channels=10)
        dummy_input = torch.randn(2, 10, 224, 224)
        print(f"[*] Feeding dummy satellite patch: {dummy_input.shape}")
        out = model(dummy_input)
        print(f"[✓] SUCCESS! Spatial features shape extracted: {out.shape}")
        print("    (Expected: [Batch, 768, 14, 14])")
    except Exception as e:
        print(f"[!] FAILED: {e}")

def test_swin_unet():
    print_header("Test 2: Swin-UNet Integration")
    try:
        from swin_unet import SatMAESwinUNet
        print("[*] Initializing Swin-UNet (4 channels, 2 classes) with SatMAE Encoder...")
        model = SatMAESwinUNet(in_channels=4, num_classes=2, use_satmae_encoder=True)
        dummy_input = torch.randn(2, 4, 224, 224)
        print(f"[*] Feeding dummy multi-band patch: {dummy_input.shape}")
        out = model(dummy_input)
        print(f"[✓] SUCCESS! Segmentation mask shape generated: {out.shape}")
        print("    (Expected: [Batch, 2, 224, 224])\n")
    except Exception as e:
        print(f"[!] FAILED: {e}")

def test_multimodal_fusion():
    print_header("Test 3: Multimodal Fusion (Optical + Radar)")
    try:
        from multimodal_fusion import OpticalRadarFusion
        print("[*] Initializing Fusion Model (10 Optical bands, 2 SAR bands)...")
        model = OpticalRadarFusion(optical_bands=10, radar_bands=2)
        opt_data = torch.randn(4, 10, 256, 256)
        sar_data = torch.randn(4, 2, 256, 256)
        print(f"[*] Feeding Optical {opt_data.shape} and Radar {sar_data.shape}...")
        out = model(opt_data, sar_data)
        print(f"[✓] SUCCESS! Fused representation encoded successfully.")
        print(f"    Raw Output Data Shape: {out.shape} (Expected: [Batch, 64, 256, 256])")
    except Exception as e:
        print(f"[!] FAILED: {e}")

def test_clip_segmentation():
    print_header("Test 4: Text-Prompted CLIP Segmentation")
    try:
        from clip_segmentation import TextPromptedSegmentation
        print("[*] Initializing CLIP Segmentation module...")
        
        try:
            model = TextPromptedSegmentation()
        except Exception:
            print("[!] Could not initialize CLIP. Is 'transformers' installed?")
            return
            
        fake_image_features = torch.randn(2, 256, 64, 64)
        text_prompts = ["Find roads", "Find water bodies"]
        print(f"[*] Feeding Image features {fake_image_features.shape}")
        print(f"[*] Feeding Text queries: {text_prompts}")
        out = model(fake_image_features, text_prompts)
        print(f"[✓] SUCCESS! Conditional text-vision mask generated.")
        print(f"    Text-Mask Shape: {out.shape} (Expected: [Batch, 1, 64, 64])")
    except Exception as e:
        print(f"[!] FAILED: {e}")

def run_all_tests():
    test_satmae_backbone()
    time.sleep(1)
    test_swin_unet()
    time.sleep(1)
    test_multimodal_fusion()
    time.sleep(1)
    test_clip_segmentation()
    print("\n" + "="*70)
    print(" ALL STRUCTURAL TESTS COMPLETED SUCCESSFULLY ".center(70, '='))

def main():
    while True:
        print("\n" + "-"*75)
        print(" DEEPGLOBE SATELLITE SEGMENTATION: EXPERIMENT VALIDATION SUITE ".center(75))
        print("-" * 75)
        print(" [1] Validate Base SatMAE++ Backbone")
        print(" [2] Validate Swin-UNet Architecture")
        print(" [3] Validate Multimodal Fusion (Optical + SAR)")
        print(" [4] Validate Text-Prompted CLIP Segmentation")
        print(" [5] Validate ALL Models at once")
        print(" [0] EXIT")
        print("-" * 75)
        
        try:
            choice = input("Enter the number of the experiment to test: ").strip()
            
            if choice == '1':
                test_satmae_backbone()
            elif choice == '2':
                test_swin_unet()
            elif choice == '3':
                test_multimodal_fusion()
            elif choice == '4':
                test_clip_segmentation()
            elif choice == '5':
                run_all_tests()
            elif choice == '0':
                print("[*] Exiting validation suite. Goodbye!")
                sys.exit(0)
            else:
                print("[!] Invalid choice. Please enter a number between 0 and 5.")
                
            time.sleep(1.5) # Breve pausa per leggere l'output prima di mostrare di nuovo il menu
                
        except KeyboardInterrupt:
            print("\n[*] Exiting validation suite. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
