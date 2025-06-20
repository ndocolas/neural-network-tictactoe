// Global variables
let trainingEventSource = null;
let trainingInProgress = false;
let currentGeneration = 0;
let totalGenerations = 0;
let bestFitness = 0.0;
let modelLoaded = false;
let trainingStartTime = null;
let trainingInterval = null;

// Make these variables globally available for the game
window.trainingState = {
    modelLoaded: false,
    trainingInProgress: false,
    bestFitness: 0.0
};

// DOM Elements
const startButton = document.getElementById('start-training');
const cancelButton = document.getElementById('cancel-training');
const trainingStatus = document.getElementById('training-status');
const progressBar = document.getElementById('training-progress');
const generationCounter = document.getElementById('generation-counter');
const bestFitnessDisplay = document.getElementById('best-fitness');
const elapsedTimeDisplay = document.getElementById('elapsed-time');
const trainingLog = document.getElementById('training-log');

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Set up training controls
    const startButton = document.getElementById('start-training');
    const cancelButton = document.getElementById('cancel-training');
    
    if (startButton) {
        startButton.addEventListener('click', startTraining);
    }
    
    if (cancelButton) {
        cancelButton.addEventListener('click', cancelTraining);
        cancelButton.disabled = true;
    }
    
    // Initialize model status
    loadModel();
});

// Function to start the training
async function startTraining() {
    if (trainingInProgress) {
        logMessage('Um treinamento já está em andamento');
        return;
    }
    const generations = parseInt(document.getElementById('generations').value) || 10;
    const gamesPerGen = parseInt(document.getElementById('games-per-gen').value) || 10;
    const populationSize = parseInt(document.getElementById('population-size').value) || 10;
    
    // Reset UI
    resetTrainingUI();
    trainingInProgress = true;
    trainingStartTime = Date.now();
    
    // Start updating elapsed time
    if (trainingInterval) clearInterval(trainingInterval);
    trainingInterval = setInterval(updateElapsedTime, 1000);
    
    // Update button states
    startButton.disabled = true;
    if (cancelButton) cancelButton.disabled = false;
    
    // Close any existing connection
    if (trainingEventSource) {
        trainingEventSource.close();
        trainingEventSource = null;
    }
    
    try {
        // Start the training process
        logMessage('Iniciando treinamento...');
        const response = await fetch(`/api/train?generations=${generations}&games=${gamesPerGen}&population=${populationSize}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.message || 'Falha ao iniciar o treinamento');
        }

        // Start SSE connection for training updates
        setupSSEConnection();
        
    } catch (error) {
        console.error('Error starting training:', error);
        logMessage(`Erro ao iniciar o treinamento: ${error.message}`);
        trainingComplete({ status: 'error', message: error.message });
    }
}

// Set up Server-Sent Events connection for training updates
function setupSSEConnection() {
    // Close any existing connection
    if (trainingEventSource) {
        trainingEventSource.close();
    }
    
    // Create a new EventSource connection to the status endpoint
    trainingEventSource = new EventSource('/api/train/status');
    
    // Handle messages from the server
    trainingEventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('Received training update:', data);
            
            // Update UI with the received data
            updateTrainingUI(data);
            
            // Log the message if present
            if (data.message) {
                logMessage(data.message);
            }
            
            // Handle completion states
            if (data.status === 'completed' || data.status === 'error' || data.status === 'cancelled') {
                trainingComplete(data);
            }
        } catch (e) {
            console.error('Error parsing training data:', e);
            logMessage('Erro ao processar dados de treinamento: ' + e.message);
        }
    };
    
    // Handle connection opened
    trainingEventSource.onopen = function() {
        console.log('SSE connection opened');
        logMessage('Conexão com o servidor de treinamento estabelecida');
    };
    
    // Handle errors
    trainingEventSource.onerror = function(error) {
        console.error('EventSource error:', error);
        if (trainingEventSource.readyState === EventSource.CLOSED) {
            logMessage('Conexão com o servidor de treinamento fechada');
            // Attempt to reconnect after a delay
            setTimeout(setupSSEConnection, 5000);
        } else {
            logMessage('Erro na conexão com o servidor de treinamento. Tentando reconectar...');
            // Attempt to reconnect after a delay
            setTimeout(setupSSEConnection, 5000);
        }
    };
    
    // Add event listener for page unload to close the connection
    window.addEventListener('beforeunload', function() {
        if (trainingEventSource) {
            trainingEventSource.close();
            trainingEventSource = null;
        }
    });
}

// Function to cancel the training
function cancelTraining() {
    if (trainingInProgress && confirm('Tem certeza que deseja cancelar o treinamento?')) {
        logMessage('Solicitando cancelamento do treinamento...');
        
        fetch('/api/train/cancel', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.message || 'Falha ao cancelar o treinamento');
                });
            }
            return response.json();
        })
        .then(data => {
            logMessage('Solicitação de cancelamento enviada com sucesso');
            // Don't complete training here - wait for the cancelled status from SSE
        })
        .catch(error => {
            console.error('Error cancelling training:', error);
            logMessage('Erro ao cancelar o treinamento: ' + error.message);
            
            // If we can't reach the server, assume training is stopped
            if (error.message.includes('Failed to fetch')) {
                logMessage('Não foi possível conectar ao servidor. O treinamento pode continuar em execução.');
            }
        });
    }
}

// Function to update the UI with training progress
function updateTrainingUI(data) {
    if (!data) return;
    
    console.log('Updating UI with data:', data);
    
    // Update progress bar
    if (data.progress !== undefined) {
        const progressPercent = Math.min(100, Math.max(0, (data.progress * 100))).toFixed(1);
        progressBar.style.width = `${progressPercent}%`;
        progressBar.setAttribute('aria-valuenow', progressPercent);
        progressBar.textContent = `${progressPercent}%`;
    }
    
    // Update generation counter
    if (data.generation !== undefined && data.total_generations !== undefined) {
        generationCounter.textContent = `Geração ${data.generation} de ${data.total_generations}`;
        currentGeneration = data.generation;
        totalGenerations = data.total_generations;
    }
    
    // Update best fitness
    if (data.best_fitness !== undefined) {
        bestFitness = parseFloat(data.best_fitness);
        bestFitnessDisplay.textContent = `Melhor Fitness: ${bestFitness.toFixed(4)}`;
    }
    
    // Update current fitness
    if (data.current_fitness !== undefined) {
        const currentFitness = parseFloat(data.current_fitness);
        document.getElementById('current-fitness').textContent = `Fitness Atual: ${currentFitness.toFixed(4)}`;
    }
    
    // Update model loaded status
    if (data.model_loaded !== undefined) {
        modelLoaded = data.model_loaded;
        updateModelStatusUI();
    }
    
    // Update training status
    if (data.status) {
        trainingStatus.textContent = data.status === 'training' ? 'Em andamento...' : 
                                    data.status === 'completed' ? 'Concluído' : 
                                    data.status === 'error' ? 'Erro' : 
                                    data.status === 'cancelled' ? 'Cancelado' : data.status;
    }
}

// Function to update elapsed time
function updateElapsedTime() {
    if (!trainingStartTime) return;
    
    const elapsed = Math.floor((Date.now() - trainingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    elapsedTimeDisplay.textContent = `Tempo decorrido: ${minutes}m ${seconds}s`;
}

// Function to handle training completion
function trainingComplete(data) {
    window.trainingState.trainingInProgress = false;
    
    // Update the global training state
    if (data.status === 'completed') {
        window.trainingState.modelLoaded = true;
        window.trainingState.bestFitness = data.best_fitness || 0.0;
        
        // Enable vs AI mode if it was disabled
        const vsAiBtn = document.querySelector('[data-mode="vs_ai"]');
        if (vsAiBtn) {
            vsAiBtn.disabled = false;
            vsAiBtn.title = '';
        }
    }
    if (!trainingInProgress && data.status !== 'completed' && data.status !== 'error' && data.status !== 'cancelled') {
        return; // Already completed
    }
    
    trainingInProgress = false;
    
    // Clear the interval for updating elapsed time
    if (trainingInterval) {
        clearInterval(trainingInterval);
        trainingInterval = null;
    }
    
    // Close the EventSource connection
    if (trainingEventSource) {
        trainingEventSource.close();
        trainingEventSource = null;
    }
    
    // Update button states
    startButton.disabled = false;
    if (cancelButton) cancelButton.disabled = true;
    
    // Update status message based on completion status
    let statusMessage = 'Treinamento concluído';
    if (data.status === 'error') {
        statusMessage = 'Erro no treinamento';
        logMessage(`Erro: ${data.message || 'Erro desconhecido durante o treinamento'}`);
    } else if (data.status === 'cancelled') {
        statusMessage = 'Treinamento cancelado';
        logMessage('Treinamento cancelado pelo usuário');
    }
    
    // Update the UI with final status
    if (data.status) {
        trainingStatus.textContent = statusMessage;
        
        // If training completed successfully, update the model status
        if (data.status === 'completed') {
            loadModel(); // Reload the model to ensure we have the latest version
        }
    }
    
    // Update status
    if (data) {
        if (data.status === 'completed') {
            logMessage('Treinamento concluído com sucesso!');
            trainingStatus.textContent = 'Concluído';
            trainingStatus.className = 'text-success';
            
            // Load the trained model
            loadModel();
        } else if (data.status === 'cancelled') {
            logMessage('Treinamento cancelado pelo usuário');
            trainingStatus.textContent = 'Cancelado';
            trainingStatus.className = 'text-warning';
        } else if (data.status === 'error') {
            logMessage(`Erro durante o treinamento: ${data.message || 'Erro desconhecido'}`);
            trainingStatus.textContent = 'Erro';
            trainingStatus.className = 'text-danger';
        }
    }
    
    // Show completion message
    if (data.status === 'completed') {
        logMessage('Treinamento concluído com sucesso!');
    } else if (data.status === 'cancelled') {
        logMessage('Treinamento cancelado pelo usuário');
    } else if (data.status === 'error') {
        logMessage(`Erro durante o treinamento: ${data.message || 'Erro desconhecido'}`);
    }
    
    // Reload the model after training completes
    loadModel();
}

// Function to reset the training UI
function resetTrainingUI() {
    progressBar.style.width = '0%';
    progressBar.setAttribute('aria-valuenow', '0');
    progressBar.textContent = '0%';
    progressBar.className = 'progress-bar progress-bar-striped progress-bar-animated';
    
    generationCounter.textContent = 'Geração 0 de 0';
    bestFitnessDisplay.textContent = 'Melhor Fitness: 0.0000';
    document.getElementById('current-fitness').textContent = 'Fitness Atual: 0.0000';
    elapsedTimeDisplay.textContent = 'Tempo decorrido: 0m 0s';
    trainingStatus.textContent = 'Não iniciado';
    trainingStatus.className = 'text-muted';
    
    // Clear the log but keep the header
    if (trainingLog) {
        trainingLog.innerHTML = '<div class="log-header">Registro de Treinamento:</div><div class="log-entries"></div>';
    }
    
    // Reset training start time
    trainingStartTime = null;
    bestFitness = 0.0;
}

// Function to log messages to the training log
function logMessage(message) {
    if (!trainingLog) return;
    
    // Make sure we have the log entries container
    let entriesContainer = trainingLog.querySelector('.log-entries');
    if (!entriesContainer) {
        entriesContainer = document.createElement('div');
        entriesContainer.className = 'log-entries';
        trainingLog.appendChild(entriesContainer);
    }
    
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    
    const timestamp = new Date().toLocaleTimeString();
    logEntry.textContent = `[${timestamp}] ${message}`;
    
    entriesContainer.appendChild(logEntry);
    entriesContainer.scrollTop = entriesContainer.scrollHeight;
}

// Function to update the model status UI
function updateModelStatusUI() {
    const modelStatusElement = document.getElementById('model-status');
    if (!modelStatusElement) return;
    
    if (window.trainingState.modelLoaded) {
        modelStatusElement.textContent = 'Carregado';
        modelStatusElement.className = 'text-success';
        
        // Show best fitness if available
        if (window.trainingState.bestFitness > 0) {
            const fitnessElement = document.getElementById('best-fitness');
            if (fitnessElement) {
                fitnessElement.textContent = `Melhor Fitness: ${window.trainingState.bestFitness.toFixed(4)}`;
            }
        }
    } else {
        modelStatusElement.textContent = 'Não carregado';
        modelStatusElement.className = 'text-danger';
    }
    const modelStatus = document.getElementById('model-status');
    if (!modelStatus) return;
    
    if (modelLoaded) {
        modelStatus.textContent = 'Modelo carregado';
        modelStatus.className = 'text-success';
    } else {
        modelStatus.textContent = 'Modelo não carregado';
        modelStatus.className = 'text-danger';
    }
}

// Function to load the model
function loadModel() {
    // This function is called when the page loads and after training completes
    fetch('/api/model/status')
        .then(response => response.json())
        .then(data => {
            window.trainingState.modelLoaded = data.loaded || false;
            window.trainingState.bestFitness = data.best_fitness || 0.0;
            
            // Update UI
            updateModelStatusUI();
            
            // Enable/disable vs AI button based on model status
            const vsAiBtn = document.querySelector('[data-mode="vs_ai"]');
            if (vsAiBtn) {
                vsAiBtn.disabled = !window.trainingState.modelLoaded;
                vsAiBtn.title = window.trainingState.modelLoaded ? '' : 'Treine um modelo primeiro';
            }
        })
        .catch(error => {
            console.error('Error checking model status:', error);
            logMessage('Erro ao verificar status do modelo');
        });
    
    // Check if we can make a neural network move (verifies model is loaded)
    fetch('/api/move/neural-network', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            board: [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Model load response:', data);
        if (data.error) {
            console.error('Error loading model:', data.error);
            modelLoaded = false;
            logMessage(`Erro ao carregar o modelo: ${data.error}`);
        } else {
            console.log('Model loaded successfully');
            modelLoaded = true;
            logMessage('Modelo carregado com sucesso!');
        }
        updateModelStatusUI();
    })
    .catch(error => {
        console.error('Error loading model:', error);
        modelLoaded = false;
        logMessage(`Erro ao carregar o modelo: ${error.message}`);
        updateModelStatusUI();
    });
}

// Initialize the game board for playing against Minimax
function initializeMinimaxGame() {
    // Add your Minimax game initialization code here
    console.log('Initializing Minimax game...');
    // This function should set up the game board and event listeners
    // for playing against the Minimax AI
}

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Load the model when the page loads
    loadModel();
    
    // Initialize the Minimax game
    initializeMinimaxGame();
    
    // Set up event listeners for training controls
    if (startButton) {
        startButton.addEventListener('click', startTraining);
    }
    
    if (cancelButton) {
        cancelButton.addEventListener('click', cancelTraining);
        cancelButton.disabled = true;
    }
});

// Export functions for use in other scripts
window.trainingModule = {
    startTraining,
    cancelTraining,
    loadModel
};
