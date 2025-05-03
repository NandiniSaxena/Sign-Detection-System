document.getElementById('startCamera').addEventListener('click', () => {
    const videoFeed = document.getElementById('videoFeed');
    const loadingSpinner = document.getElementById('loadingSpinner');
    loadingSpinner.classList.remove('hidden');
    videoFeed.src = '/video_feed';
    videoFeed.onload = () => loadingSpinner.classList.add('hidden');
    videoFeed.onerror = () => {
        loadingSpinner.classList.add('hidden');
        alert('Failed to load camera feed. Please check camera permissions.');
    };
});

document.getElementById('stopCamera').addEventListener('click', () => {
    fetch('/stop_camera')
        .then(response => response.json())
        .then(data => {
            document.getElementById('videoFeed').src = '';
            document.getElementById('digit').textContent = 'Digit: None';
            document.getElementById('confidence').textContent = 'Confidence: 0.00%';
            document.getElementById('description').textContent = 'Gesture: None';
        })
        .catch(error => alert('Error stopping camera: ' + error));
});

document.getElementById('showMatrix').addEventListener('click', () => {
    document.getElementById('matrixModal').classList.remove('hidden');
});

document.getElementById('closeMatrix').addEventListener('click', () => {
    document.getElementById('matrixModal').classList.add('hidden');
});

document.getElementById('retrainModel').addEventListener('click', () => {
    if (confirm('Retraining the model may take a few minutes. Proceed?')) {
        const progress = document.getElementById('retrainProgress');
        progress.classList.remove('hidden');
        fetch('/retrain_model')
            .then(response => response.json())
            .then(data => {
                progress.classList.add('hidden');
                alert(data.status);
            })
            .catch(error => {
                progress.classList.add('hidden');
                alert('Error retraining model: ' + error);
            });
    }
});

document.getElementById('viewLog').addEventListener('click', () => {
    window.open('/log', '_blank');
});

document.getElementById('darkModeToggle').addEventListener('click', () => {
    document.body.classList.toggle('dark');
    const sunIcon = document.getElementById('sunIcon');
    const moonIcon = document.getElementById('moonIcon');
    sunIcon.classList.toggle('hidden');
    moonIcon.classList.toggle('hidden');
    localStorage.setItem('darkMode', document.body.classList.contains('dark'));
});

// Load dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark');
    document.getElementById('sunIcon').classList.remove('hidden');
    document.getElementById('moonIcon').classList.add('hidden');
}

function updatePredictions() {
    fetch('/predictions')
        .then(response => response.json())
        .then(data => {
            document.getElementById('digit').textContent = `Digit: ${data.digit}`;
            document.getElementById('confidence').textContent = `Confidence: ${data.confidence}`;
            document.getElementById('description').textContent = `Gesture: ${data.description}`;
        })
        .catch(error => console.error('Error fetching predictions:', error));
}

setInterval(updatePredictions, 1000); // Update every second
