document.getElementById('generate-form').addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData(event.target);

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '';

        if (data.image_paths) {
            data.image_paths.forEach(path => {
                const img = document.createElement('img');
                img.src = '/' + path;  // Flask static path
                img.alt = 'Generated Image';
                img.style.display = 'block';
                img.style.marginBottom = '10px';
                img.style.maxWidth = '100%';
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

// Update default checkpoint text when dataset changes
document.getElementById('dataset').addEventListener('change', (event) => {
    const dataset = event.target.value;
    const defaultCheckpointDiv = document.getElementById('default-checkpoint');
    if (dataset === "celeba") {
        defaultCheckpointDiv.textContent = "Default: celeba checkpoint";
    } else if (dataset === "flowers") {
        defaultCheckpointDiv.textContent = "Default: flowers checkpoint";
    }
});
