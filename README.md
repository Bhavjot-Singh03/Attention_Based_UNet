# Attention_Based_UNet

The attention based unet architecture is designed specifically to achieve the highest mean intersection over union on the BraTS 2020 dataset. This architecture incorporates a lightweight variant of the Convolutional Block Attention Module (CBAM) in encoder skip connections and utilizes residual blocks for better contextual information. This approach attains an outstanding mIoU score of 0.9720, displaying remarkable accuracy in capturing complex anatomical structures.

## Architecture

### 1. Model Architecture
![attention_unet](https://github.com/Bhavjot-Singh03/Attention_Based_UNet/assets/131793243/c353bb87-22c1-4928-87c0-293b8ba7ecaf)

### 2. Attention Mechanism
![Attention_Mechanism](https://github.com/Bhavjot-Singh03/Attention_Based_UNet/assets/131793243/3f745383-cb91-4501-8d0a-def8446d4825)

## Performance Comparison

| Model               | mIoU   | Accuracy | Precision | Recall  | Specificity |
|---------------------|--------|----------|-----------|---------|-------------|
| Enhanced U-Net      | 0.8935 | 0.9980   | 0.9973    | 0.9970  | 0.9983      |
| U-Net               | 0.8326 | 0.9939   | 0.9941    | 0.9927  | 0.9980      |
| DeepLabv3+          | 0.93   | 0.9967   | -         | -       | -           |
| EfficientNet FPN    | 0.9083 | -        | -         | -       | -           |
| Att_B_Unet          | 0.9720 | 0.9782   | 0.9782    | 0.9782  | 0.9927      |

