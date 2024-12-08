from flask import Flask, render_template, request, jsonify
from pathlib import Path

from diffuse_generator import generate_images

app = Flask(__name__)

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
            checkpoint_path = Path("uploads/checkpoints") / checkpoint_file.filename
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_file.save(checkpoint_path)
        else:
            checkpoint_path = DEFAULT_CHECKPOINTS.get(dataset_type)

        # Get other parameters
        num_steps = int(request.form.get("num_steps", 1000))
        seed = int(request.form.get("seed", 42))
        num_images = int(request.form.get("num_images", 1))

        # Generate images
        image_paths = generate_images(
            checkpoint=checkpoint_path,
            num_steps=num_steps,
            seed=seed,
            num_images=num_images
        )

        return jsonify({"image_paths": [str(path) for path in image_paths]})
    except Exception as e:
        app.logger.error(f"Error during image generation: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5005)
