from flask import Flask, render_template, request, jsonify
from pathlib import Path

from diffuse_generator import generate_images

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate images based on the parameters."""
    try:
        # Get parameters from the form data
        num_steps = int(request.form.get("num_steps", 50))
        seed = int(request.form.get("seed", 42))
        num_images = int(request.form.get("num_images", 1))

        # Handle uploaded checkpoint file
        checkpoint_file = request.files['checkpoint']
        checkpoint_path = Path("uploads/checkpoints") / checkpoint_file.filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file.save(checkpoint_path)

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
    app.run(debug=True, port=5004)
