from flask import Flask, render_template, request, jsonify
from pathlib import Path
import os
import uuid
from diffuse_generator import generate_images

app = Flask(__name__, static_folder="static")

# Default checkpoints for each dataset
DEFAULT_CHECKPOINTS = {
    "celeba": "/workspace/outputs/celeba/diffusion_model_celeba.ckpt",
    "flowers": "/workspace/outputs/flowers/diffusion_model_flowers.ckpt"
}

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate images based on the parameters."""
    try:
        # Get dataset type
        dataset_type = request.form.get('dataset', 'celeba').lower()  # Default to celeba

        # Handle checkpoint file
        checkpoint_file = request.files.get('checkpoint')
        checkpoint_path = None

        # Use uploaded checkpoint if provided, otherwise use default
        if checkpoint_file:
            checkpoint_path = Path("static/uploads/checkpoints") / checkpoint_file.filename
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file.save(checkpoint_path)
        else:
            checkpoint_path = DEFAULT_CHECKPOINTS.get(dataset_type)

        # Validate checkpoint path
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return jsonify({"error": f"{dataset_type.capitalize()} model is missing. Please upload it or ensure it exists."}), 400

        # Get other parameters
        num_steps = int(request.form.get("num_steps", 50))
        seed = int(request.form.get("seed", 42))
        num_images = int(request.form.get("num_images", 1))

        # Generate a unique identifier for this request
        unique_id = uuid.uuid4().hex

        # Generate images
        image_paths = generate_images(
            checkpoint=checkpoint_path,
            num_steps=num_steps,
            seed=seed,
            num_images=num_images,
            unique_id=unique_id  # Pass the unique identifier
        )

        # Generate unique IDs for the image paths to avoid cache issues
        image_paths_with_ids = [
            f"{path}?uid={uuid.uuid4().hex}" for path in image_paths
        ]

        return jsonify({"image_paths": image_paths_with_ids})
    except Exception as e:
        app.logger.error(f"Error during image generation: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5005)
