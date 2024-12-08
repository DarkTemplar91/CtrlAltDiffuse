// Function to set a random seed on page load
function setRandomSeed() {
    const seedInput = document.getElementById('seed');
    seedInput.value = Math.floor(Math.random() * 10000); // Random value between 0-9999
}

// Set a random seed when the page loads
document.addEventListener('DOMContentLoaded', setRandomSeed);

document.getElementById('generate-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData(event.target);

    try {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = ''; // Clear previous results

        // Show loading spinner
        const spinner = document.createElement('div');
        spinner.className = 'spinner-border text-primary';
        spinner.role = 'status';
        spinner.innerHTML = '<span class="visually-hidden">Loading...</span>';
        resultsDiv.appendChild(spinner);

        // Send form data to the server
        const response = await fetch('/generate', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        resultsDiv.innerHTML = ''; // Clear spinner

        if (data.image_paths) {
            data.image_paths.forEach(path => {
                const img = document.createElement('img');
                img.src = '/' + path; // Flask static path
                img.alt = 'Generated Image';
                img.style.display = 'block';
                img.style.marginBottom = '10px';
                resultsDiv.appendChild(img);
            });
        } else {
            resultsDiv.innerText = 'No images generated.';
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('results').innerText = `Error: ${error.message}`;
    }
});

// Handle dataset changes to update the default checkpoint message
document.getElementById('dataset').addEventListener('change', (event) => {
    const dataset = event.target.value;
    const defaultCheckpointDiv = document.getElementById('default-checkpoint');
    if (dataset === "celeba") {
        defaultCheckpointDiv.textContent = "Default: CelebA checkpoint";
    } else if (dataset === "flowers") {
        defaultCheckpointDiv.textContent = "Default: Flowers102 checkpoint";
    }
});

// Handle model download
document.getElementById('download-model').addEventListener('click', async () => {
    const dataset = document.getElementById('dataset').value;

    try {
        const response = await fetch(`/download?dataset=${dataset}`, {
            method: 'GET',
        });

        if (!response.ok) {
            throw new Error(`Failed to download model for ${dataset}`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        // Create a hidden link to trigger download
        const a = document.createElement('a');
        a.href = url;
        a.download = `${dataset}_model.ckpt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}`);
    }
});
