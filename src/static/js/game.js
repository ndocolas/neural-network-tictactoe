document.addEventListener('DOMContentLoaded', () => {
    // Elementos da interface
    const boardElement = document.getElementById('board');
    const statusElement = document.getElementById('gameStatus');
    const resetBtn = document.getElementById('resetBtn');
    const backBtn = document.getElementById('backBtn');
    const gameContainer = document.getElementById('gameContainer');
    const gameOptions = document.getElementById('gameOptions');
    const trainingModal = document.getElementById('trainingModal');
    
    // Botões de modo de jogo
    const playMinimaxBtn = document.getElementById('playMinimaxBtn');
    const trainNetworkBtn = document.getElementById('trainNetworkBtn');
    const playNetworkBtn = document.getElementById('playNetworkBtn');
    
    // Estado do jogo
    let gameMode = null; // 'minimax' ou 'neural_network'
    let board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ];
    let gameOver = false;
    let winner = null;
    
    // Inicializa o jogo
    function initializeGame(mode) {
        gameMode = mode;
        gameOver = false;
        winner = null;
        board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ];
        
        // Atualiza a interface
        updateBoard();
        updateStatus('Sua vez! Clique em uma célula para começar.');
        
        // Mostra o jogo e esconde as opções
        gameOptions.style.display = 'none';
        gameContainer.style.display = 'block';
        
        // Se for modo rede neural, desabilita os botões até o treinamento
        if (mode === 'neural_network') {
            updateStatus('Carregando rede neural...');
            // Aqui você carregaria o modelo treinado
            setTimeout(() => {
                updateStatus('Sua vez! Clique em uma célula para começar.');
            }, 1000);
        }
    }
    
    // Inicializa o tabuleiro
    function initializeBoard() {
        boardElement.innerHTML = '';
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = i;
                cell.dataset.col = j;
                cell.addEventListener('click', handleCellClick);
                boardElement.appendChild(cell);
            }
        }
    }

    // Atualiza a exibição do tabuleiro
    function updateBoard() {
        const cells = document.querySelectorAll('.cell');
        cells.forEach((cell, index) => {
            const row = Math.floor(index / 3);
            const col = index % 3;
            const value = board[row][col];
            
            cell.textContent = value === 1 ? 'X' : value === -1 ? 'O' : '';
            cell.className = 'cell';
            
            if (value === 1) cell.classList.add('x');
            else if (value === -1) cell.classList.add('o');
            
            // Desabilita células já preenchidas
            if (value !== 0 || gameOver) {
                cell.style.pointerEvents = 'none';
            } else {
                cell.style.pointerEvents = 'auto';
            }
        });
    }
    
    // Atualiza a mensagem de status
    function updateStatus(message) {
        statusElement.textContent = message;
    }
    
    // Verifica se há um vencedor
    function checkWinner() {
        // Verifica linhas
        for (let i = 0; i < 3; i++) {
            if (board[i][0] !== 0 && board[i][0] === board[i][1] && board[i][1] === board[i][2]) {
                return board[i][0];
            }
        }
        
        // Verifica colunas
        for (let j = 0; j < 3; j++) {
            if (board[0][j] !== 0 && board[0][j] === board[1][j] && board[1][j] === board[2][j]) {
                return board[0][j];
            }
        }
        
        // Verifica diagonais
        if (board[0][0] !== 0 && board[0][0] === board[1][1] && board[1][1] === board[2][2]) {
            return board[0][0];
        }
        if (board[0][2] !== 0 && board[0][2] === board[1][1] && board[1][1] === board[2][0]) {
            return board[0][2];
        }
        
        // Verifica empate
        if (board.every(row => row.every(cell => cell !== 0))) {
            return 'draw';
        }
        
        return null;
    }
    
    // Finaliza o jogo
    function endGame(winner) {
        gameOver = true;
        let message = '';
        
        if (winner === 'draw') {
            message = 'Empate!';
        } else if (winner === 1) {
            message = 'A IA venceu!';
        } else {
            message = 'Você venceu!';
        }
        
        updateStatus(message);
    }

    // Manipula o clique em uma célula
    async function handleCellClick(e) {
        if (gameOver) return;
        
        const row = parseInt(e.target.dataset.row);
        const col = parseInt(e.target.dataset.col);
        
        // Verifica se a célula está vazia
        if (board[row][col] !== 0) return;
        
        // Faz a jogada do jogador
        board[row][col] = -1; // -1 representa o jogador
        updateBoard();
        
        // Verifica se o jogador venceu
        winner = checkWinner();
        if (winner) {
            endGame(winner);
            return;
        }
        
        // Faz a jogada da IA
        updateStatus('A IA está pensando...');
        await makeAIMove();
        
        // Verifica se a IA venceu
        winner = checkWinner();
        if (winner) {
            endGame(winner);
        } else {
            updateStatus('Sua vez!');
        }
    }

    // Realiza a jogada da IA
    async function makeAIMove() {
        try {
            let response;
            
            if (gameMode === 'minimax') {
                // Chama o endpoint para obter a jogada do Minimax
                response = await fetch('/api/move/minimax', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ board })
                });
            } else if (gameMode === 'neural_network') {
                // Chama o endpoint para obter a jogada da rede neural
                response = await fetch('/api/move/neural-network', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ board })
                });
            }
            
            if (!response.ok) {
                throw new Error('Erro ao obter jogada da IA');
            }
            
            const data = await response.json();
            
            // Atualiza o tabuleiro com a jogada da IA
            if (data.row !== undefined && data.col !== undefined) {
                board[data.row][data.col] = 1; // 1 representa a IA
                updateBoard();
            }
            
        } catch (error) {
            console.error('Erro ao fazer jogada da IA:', error);
            updateStatus('Erro ao conectar com a IA. Tente novamente.');
        }
    }
    
    // Reseta o jogo
    function resetGame() {
        board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ];
        gameOver = false;
        winner = null;
        updateBoard();
        updateStatus('Sua vez! Clique em uma célula para começar.');
    }
    
    // Volta para o menu principal
    function backToMenu() {
        gameOptions.style.display = 'block';
        gameContainer.style.display = 'none';
    }

    // Inicializa o jogo quando a página carregar
    initializeBoard();
    
    // Event Listeners
    resetBtn.addEventListener('click', resetGame);
    backBtn.addEventListener('click', backToMenu);
    
    // Event listeners para os botões de modo de jogo
    playMinimaxBtn.addEventListener('click', () => initializeGame('minimax'));
    trainNetworkBtn.addEventListener('click', () => trainingModal.style.display = 'flex');
    playNetworkBtn.addEventListener('click', () => initializeGame('neural_network'));
    
    // Inicializa o modal de treinamento
    document.getElementById('cancelTraining').addEventListener('click', () => {
        trainingModal.style.display = 'none';
    });
    
    document.getElementById('startTraining').addEventListener('click', async () => {
        const generations = parseInt(document.getElementById('generations').value);
        const games = parseInt(document.getElementById('games').value);
        const population = parseInt(document.getElementById('population').value);
        
        // Inicia o treinamento (isso será implementado no training.js)
        startTraining(generations, games, population);
        trainingModal.style.display = 'none';
    });
    initializeBoard();
    resetGame();
});
