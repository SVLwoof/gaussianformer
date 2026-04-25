import os
from gaussianformer.models.config import GaussianFormerConfig
from gaussianformer.models.gaussianformer import GaussianFormer


def main():
    """
    This script initializes a new GaussianFormer model with a default configuration
    and saves it to a local directory. This allows you to test the inference
    pipeline without needing a fully trained model or a Hugging Face repository.
    """
    # Define the directory where the model will be saved
    local_model_path = "./my-gaussianformer-model"
    os.makedirs(local_model_path, exist_ok=True)

    print(f"Creating a new GaussianFormer model configuration.")
    # You can customize the configuration here if needed
    config = GaussianFormerConfig()

    print(f"Initializing GaussianFormer with random weights.")
    # This creates an instance of your model with the specified config.
    # The weights will be randomly initialized.
    model = GaussianFormer(config)

    print(f"Saving model and configuration to: {local_model_path}")
    # The .save_pretrained() method saves two key files:
    # 1. config.json - The model's architecture and parameters.
    # 2. pytorch_model.bin - The model's weights (currently random).
    model.save_pretrained(local_model_path)

    print("\nModel saved successfully!")
    print("You can now run the inference script using the following argument:")
    print(f"  --model_id {local_model_path}")


if __name__ == '__main__':
    main()
