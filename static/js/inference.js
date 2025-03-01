let model = null;
let labels = null;

async function loadModel() {
    try {
        console.log('Starting model load from /model/model.json');

        // Load metadata first
        const metadataResponse = await fetch('/model/metadata.json');
        if (!metadataResponse.ok) {
            throw new Error('Failed to load metadata.json');
        }
        const metadata = await metadataResponse.json();
        labels = metadata.labels;
        console.log('Loaded labels:', labels);

        // Load the model
        model = await tf.loadLayersModel('/model/model.json');
        console.log('Model loaded successfully:', model);
        document.getElementById('model-status').className = 'alert alert-success';
        document.getElementById('model-status').innerHTML = '<i class="bi bi-check-circle me-2"></i>Model loaded successfully!';
        document.getElementById('input-container').style.display = 'block';

        // Add event listener for image preview
        document.getElementById('image-input').addEventListener('change', previewImage);
    } catch (error) {
        console.error('Error loading model:', error);
        document.getElementById('model-status').className = 'alert alert-danger';
        document.getElementById('model-status').innerHTML = `<i class="bi bi-exclamation-triangle me-2"></i>Error loading model: ${error.message}`;
    }
}

function previewImage(event) {
    const preview = document.getElementById('preview-image');
    const file = event.target.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
}

async function preprocessImage(imageElement) {
    // Convert the image to a tensor and preprocess it
    return tf.tidy(() => {
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeNearestNeighbor([224, 224]) // Resize to model's expected size
            .toFloat()
            .expandDims();

        // Normalize the image between -1 and 1
        return tensor.div(127.5).sub(1);
    });
}

async function runInference() {
    if (!model) {
        alert('Model not loaded yet!');
        return;
    }

    const imageInput = document.getElementById('image-input');
    const previewImage = document.getElementById('preview-image');
    const resultContainer = document.getElementById('result-container');
    const inferenceResult = document.getElementById('inference-result');

    if (!imageInput.files[0]) {
        alert('Please select an image first!');
        return;
    }

    try {
        // Show loading state
        inferenceResult.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Analyzing...</span></div><p class="mt-2">Analyzing image...</p></div>';
        resultContainer.style.display = 'block';

        console.log('Running inference on image');
        const inputTensor = await preprocessImage(previewImage);

        // Run inference
        const predictions = await model.predict(inputTensor);
        const probabilities = await predictions.data();
        console.log('Inference result:', probabilities);

        // Format results with labels
        const results = labels.map((label, index) => ({
            label: label,
            probability: (probabilities[index] * 100).toFixed(2)
        }));

        // Sort results by probability (highest first)
        results.sort((a, b) => b.probability - a.probability);

        // Create a formatted HTML result
        const resultHTML = results.map(result => `
            <div class="mb-3">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="fw-bold">${result.label}</span>
                    <span class="badge bg-primary">${result.probability}%</span>
                </div>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${result.probability}%" 
                         aria-valuenow="${result.probability}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                    </div>
                </div>
            </div>
        `).join('');

        inferenceResult.innerHTML = resultHTML;

        // Cleanup
        inputTensor.dispose();
        predictions.dispose();
    } catch (error) {
        console.error('Error during inference:', error);
        inferenceResult.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                Error during inference: ${error.message}
            </div>`;
    }
}

// Load model when page loads
window.addEventListener('load', loadModel);