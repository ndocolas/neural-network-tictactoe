// Game state
const gameState = {
    board: [
        ['', '', ''],
        ['', '', ''],
        ['', '']
    ],
    currentPlayer: 'X',
    gameMode: 'minimax', // 'minimax', 'training', 'vs_ai'
    gameOver: false,
    trainingMode: false,
    vsAIMode: false
};

// DOM Elements
const boardElement = document.getElementById('game-board');
const statusElement = document.getElementById('game-status');
const gameModes = document.querySelectorAll('.game-mode-btn');

// Initialize the game board
function initializeGameBoard() {
    if (!boardElement) return;
    
    // Create the game board HTML
    boardElement.innerHTML = '';
    for (let i = 0; i < 3; i++) {
        const row = document.createElement('div');
        row.className = 'board-row';
        
        for (let j = 0; j < 3; j++) {
            const cell = document.createElement('div');
            cell.className = 'board-cell';
            cell.dataset.row = i;
            cell.dataset.col = j;
            cell.addEventListener('click', () => handleCellClick(i, j));
            row.appendChild(cell);
        }
        
        boardElement.appendChild(row);
    }
    
    updateStatus('Escolha um modo de jogo');
}

// Handle cell click
function handleCellClick(row, col) {
    if (gameState.gameOver || gameState.board[row][col] !== '') return;
    
    // Make player move
    makeMove(row, col, 'X');
    
    // If game is not over, make AI move based on game mode
    if (!gameState.gameOver) {
        if (gameState.gameMode === 'minimax') {
            makeMinimaxMove();
        } else if (gameState.gameMode === 'vs_ai' && gameState.vsAIMode) {
            makeAIMove();
        }
    }
}

// Make a move on the board
function makeMove(row, col, player) {
    gameState.board[row][col] = player;
    updateBoard();
    
    if (checkWin(player)) {
        gameState.gameOver = true;
        updateStatus(`Jogador ${player} venceu!`);
        return true;
    }
    
    if (checkDraw()) {
        gameState.gameOver = true;
        updateStatus('Empate!');
        return true;
    }
    
    gameState.currentPlayer = player === 'X' ? 'O' : 'X';
    return false;
}

// Check for a win
function checkWin(player) {
    const b = gameState.board;
    
    // Check rows, columns and diagonals
    for (let i = 0; i < 3; i++) {
        if (b[i][0] === player && b[i][1] === player && b[i][2] === player) return true;
        if (b[0][i] === player && b[1][i] === player && b[2][i] === player) return true;
    }
    
    if (b[0][0] === player && b[1][1] === player && b[2][2] === player) return true;
    if (b[0][2] === player && b[1][1] === player && b[2][0] === player) return true;
    
    return false;
}

// Check for a draw
function checkDraw() {
    return gameState.board.every(row => row.every(cell => cell !== ''));
}

// Update the board UI
function updateBoard() {
    if (!boardElement) return;
    
    const cells = boardElement.querySelectorAll('.board-cell');
    cells.forEach((cell, index) => {
        const row = Math.floor(index / 3);
        const col = index % 3;
        cell.textContent = gameState.board[row][col];
    });
}

// Update game status
function updateStatus(message) {
    if (statusElement) {
        statusElement.textContent = message;
    }
}

// Reset the game
function resetGame() {
    gameState.board = [
        ['', '', ''],
        ['', '', ''],
        ['', '', '']
    ];
    gameState.currentPlayer = 'X';
    gameState.gameOver = false;
    
    updateBoard();
    updateStatus(`Vez do jogador ${gameState.currentPlayer}`);
}

// Set game mode
function setGameMode(mode) {
    gameState.gameMode = mode;
    gameState.trainingMode = mode === 'training';
    gameState.vsAIMode = mode === 'vs_ai';
    
    // Update active button
    gameModes.forEach(btn => {
        if (btn.dataset.mode === mode) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    resetGame();
}

// Make AI move using the trained model
async function makeAIMove() {
    try {
        const response = await fetch('/api/move/neural-network', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                board: gameState.board,
                player: 'O'
            })
        });
        
        if (!response.ok) {
            throw new Error('Erro ao obter jogada da IA');
        }
        
        const data = await response.json();
        if (data.move) {
            makeMove(data.move.row, data.move.col, 'O');
        }
    } catch (error) {
        console.error('Error making AI move:', error);
        updateStatus('Erro ao fazer jogada da IA');
    }
}

// Make Minimax move (for training mode)
function makeMinimaxMove() {
    // This would be implemented to use the minimax algorithm
    // For now, it makes a random move
    const availableMoves = [];
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            if (gameState.board[i][j] === '') {
                availableMoves.push({row: i, col: j});
            }
        }
    }
    
    if (availableMoves.length > 0) {
        const randomMove = availableMoves[Math.floor(Math.random() * availableMoves.length)];
        makeMove(randomMove.row, randomMove.col, 'O');
    }
}

// Initialize game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeGameBoard();
    
    // Set up game mode buttons
    gameModes.forEach(btn => {
        btn.addEventListener('click', () => {
            setGameMode(btn.dataset.mode);
        });
    });
});

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        gameState,
        makeMove,
        checkWin,
        checkDraw,
        makeAIMove,
        makeMinimaxMove
    };
}
