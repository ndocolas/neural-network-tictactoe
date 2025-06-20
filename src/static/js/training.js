// Inicia o treinamento da rede neural
async function startTraining(generations, games, population) {
    const trainingModal = document.getElementById('trainingModal');
    const trainingProgress = document.getElementById('trainingProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const startTrainingBtn = document.getElementById('startTraining');
    const cancelTrainingBtn = document.getElementById('cancelTraining');
    
    // Ensure progress elements exist
    let progressContent = document.getElementById('trainingProgressContent');
    let statusMessage = document.querySelector('.status-message');
    let completionNotice = document.getElementById('completionNotice');
    
    // Create progress UI if it doesn't exist
    if (!progressContent) {
        trainingProgress.innerHTML = `
            <div id="trainingProgressContent">
                <div class="progress-container">
                    <div class="progress-header">
                        <span>Progresso do Treinamento</span>
                        <span id="progressPercentage">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress" id="progressBar"></div>
                    </div>
                </div>
                
                <div class="progress-stats" id="trainingStats">
                    <div class="stat-box">
                        <div class="label">Geração Atual</div>
                        <div class="value" id="currentGeneration">0 / ${generations}</div>
                    </div>
                    <div class="stat-box">
                        <div class="label">Melhor Fitness</div>
                        <div class="value" id="bestFitness">0.0000</div>
                    </div>
                    <div class="stat-box">
                        <div class="label">Tempo Estimado</div>
                        <div class="value" id="timeRemaining">Calculando...</div>
                    </div>
                </div>
                
                <div class="progress-text">
                    <div class="status-message">
                        <span class="spinner"></span>
                        <span>Iniciando treinamento...</span>
                    </div>
                </div>
                
                <div id="completionNotice" class="completion-notice" style="display: none;">
                    <span class="icon">✓</span>
                    <div>
                        <strong>Treinamento concluído com sucesso!</strong>
                        <div id="completionDetails" class="completion-details"></div>
                    </div>
                </div>
            </div>
        `;
        
        // Update references to the newly created elements
        progressContent = document.getElementById('trainingProgressContent');
        statusMessage = document.querySelector('.status-message');
        completionNotice = document.getElementById('completionNotice');
    } else {
        // Ensure status message exists in the static HTML
        if (!statusMessage) {
            const progressText = document.querySelector('.progress-text');
            if (progressText) {
                progressText.innerHTML = `
                    <div class="status-message">
                        <span class="spinner"></span>
                        <span>Iniciando treinamento...</span>
                    </div>
                `;
                statusMessage = progressText.querySelector('.status-message');
            }
        }
    }
    
    // Show progress section
    trainingProgress.style.display = 'block';
    
    // Reset state
    if (progressBar) progressBar.style.width = '0%';
    
    const progressPercentage = document.getElementById('progressPercentage');
    const currentGeneration = document.getElementById('currentGeneration');
    const bestFitness = document.getElementById('bestFitness');
    const timeRemaining = document.getElementById('timeRemaining');
    
    if (progressPercentage) progressPercentage.textContent = '0%';
    if (currentGeneration) currentGeneration.textContent = `0 / ${generations}`;
    if (bestFitness) bestFitness.textContent = '0.0000';
    if (timeRemaining) timeRemaining.textContent = 'Calculando...';
    if (completionNotice) completionNotice.style.display = 'none';
    
    if (statusMessage) {
        statusMessage.innerHTML = `
            <span class="spinner"></span>
            <span>Iniciando treinamento...</span>
        `;
    }
    
    // Desabilita os botões durante o treinamento
    startTrainingBtn.disabled = true;
    cancelTrainingBtn.disabled = false;
    
    // Variáveis para cálculo do tempo restante
    let startTime = Date.now();
    let lastUpdateTime = startTime;
    let lastProgress = 0;
    
    try {
        // Faz a requisição para iniciar o treinamento
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                generations,
                games,
                population_size: population
            })
        });
        
        if (!response.ok) {
            throw new Error('Erro ao iniciar o treinamento');
        }
        
        // Usa Server-Sent Events para receber atualizações em tempo real
        const eventSource = new EventSource('/api/train/status');
        
        // Função para formatar o tempo restante
        const formatTime = (seconds) => {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.floor(seconds % 60);
            return `${minutes}m ${remainingSeconds}s`;
        };
        
        // Função para atualizar o tempo restante
        const updateTimeRemaining = (progress) => {
            const now = Date.now();
            const elapsed = (now - startTime) / 1000; // em segundos
            
            if (progress > 0 && progress > lastProgress) {
                const timePerPercent = elapsed / (progress * 100); // segundos por 1%
                const remainingPercent = 100 - (progress * 100);
                const remainingTime = timePerPercent * remainingPercent;
                
                document.getElementById('timeRemaining').textContent = formatTime(remainingTime);
                lastUpdateTime = now;
                lastProgress = progress;
            }
        };
        
        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.status === 'training') {
                    const { generation, progress, best_fitness } = data;
                    const percentage = Math.round(progress * 100);
                    
                    // Atualiza a barra de progresso
                    progressBar.style.width = `${percentage}%`;
                    document.getElementById('progressPercentage').textContent = `${percentage}%`;
                    document.getElementById('currentGeneration').textContent = `${generation} / ${generations}`;
                    document.getElementById('bestFitness').textContent = best_fitness ? best_fitness.toFixed(4) : '0.0000';
                    
                    // Atualiza o tempo restante
                    updateTimeRemaining(progress);
                    
                    // Atualiza a mensagem de status
                    statusMessage.innerHTML = `
                        <span class="spinner"></span>
                        <span>Treinando geração ${generation} de ${generations}...</span>
                    `;
                    
                } else if (data.status === 'completed') {
                    // Treinamento concluído com sucesso
                    eventSource.close();
                    progressBar.style.width = '100%';
                    document.getElementById('progressPercentage').textContent = '100%';
                    document.getElementById('timeRemaining').textContent = 'Concluído';
                    
                    // Mostra a notificação de conclusão
                    completionNotice.style.display = 'flex';
                    document.getElementById('completionDetails').innerHTML = `
                        <div>Gerações: ${generations}</div>
                        <div>Jogos por geração: ${games}</div>
                        <div>Melhor fitness: ${data.best_fitness ? data.best_fitness.toFixed(4) : 'N/A'}</div>
                    `;
                    
                    // Atualiza o botão de treinar novamente
                    startTrainingBtn.textContent = 'Treinar Novamente';
                    startTrainingBtn.disabled = false;
                    cancelTrainingBtn.textContent = 'Fechar';
                    
                    // Atualiza o botão de jogar contra a rede neural
                    const playNetworkBtn = document.getElementById('playNetworkBtn');
                    playNetworkBtn.disabled = false;
                    playNetworkBtn.classList.add('pulse');
                    setTimeout(() => playNetworkBtn.classList.remove('pulse'), 2000);
                    
                    // Rola para a notificação de conclusão
                    completionNotice.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    
                } else if (data.status === 'error') {
                    throw new Error(data.message || 'Erro durante o treinamento');
                }
            } catch (error) {
                console.error('Erro ao processar mensagem do servidor:', error);
                throw error;
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('Erro na conexão com o servidor:', error);
            eventSource.close();
            throw new Error('Erro na conexão com o servidor. Atualize a página e tente novamente.');
        };
        
        // Adiciona evento de cancelamento
        cancelTrainingBtn.onclick = () => {
            if (confirm('Deseja realmente cancelar o treinamento? O progresso atual será perdido.')) {
                eventSource.close();
                trainingProgress.style.display = 'none';
                startTrainingBtn.disabled = false;
                cancelTrainingBtn.disabled = false;
                
                // Envia uma requisição para cancelar o treinamento no servidor
                fetch('/api/train/cancel', { method: 'POST' })
                    .catch(err => console.error('Erro ao cancelar treinamento:', err));
            }
        };
        
    } catch (error) {
        console.error('Erro durante o treinamento:', error);
        
        // Mostra mensagem de erro
        const errorMessage = document.createElement('div');
        errorMessage.className = 'error-message';
        errorMessage.innerHTML = `
            <div style="color: #e74c3c; background-color: #fde8e8; padding: 10px 15px; border-radius: 5px; margin-top: 10px;">
                <strong>Erro durante o treinamento:</strong><br>
                ${error.message}
            </div>
        `;
        
        progressText.appendChild(errorMessage);
        startTrainingBtn.disabled = false;
        cancelTrainingBtn.disabled = false;
        
        // Rola para a mensagem de erro
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Fecha o modal de treinamento
function closeTrainingModal() {
    const trainingModal = document.getElementById('trainingModal');
    const trainingProgress = document.getElementById('trainingProgress');
    const progressBar = document.getElementById('progressBar');
    const startTrainingBtn = document.getElementById('startTraining');
    
    trainingModal.style.display = 'none';
    trainingProgress.style.display = 'none';
    progressBar.style.width = '0%';
    startTrainingBtn.disabled = false;
}

// Event listener para o botão de cancelar treinamento
document.getElementById('cancelTraining').addEventListener('click', closeTrainingModal);
