# Jogo da Velha com IA

Este é um jogo da velha (Tic-Tac-Toe) que utiliza diferentes algoritmos de IA para jogar contra o usuário, incluindo Minimax e uma rede neural treinada com algoritmo genético.

## Funcionalidades

- Jogue contra o algoritmo Minimax
- Treine sua própria rede neural usando algoritmo genético
- Jogue contra a rede neural treinada
- Interface amigável com feedback visual
- Treinamento em tempo real com atualizações de progresso

## Requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/jogo-da-velha-ia.git
   cd jogo-da-velha-ia
   ```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Como executar

1. Inicie o servidor Flask:
   ```bash
   python app.py
   ```

2. Abra seu navegador e acesse:
   ```
   http://localhost:5001
   ```

## Como jogar

1. **Jogar contra Minimax**:
   - Clique em "Jogar contra Minimax"
   - Você joga como 'O' e o computador como 'X'
   - Clique em qualquer célula vazia para fazer sua jogada

2. **Treinar Rede Neural**:
   - Clique em "Treinar Rede Neural"
   - Configure os parâmetros de treinamento (opcional)
   - Clique em "Iniciar Treinamento"
   - Acompanhe o progresso em tempo real
   - Após o treinamento, você pode jogar contra a rede neural

3. **Jogar contra Rede Neural**:
   - Após o treinamento, clique em "Jogar contra Rede Neural"
   - Jogue contra a rede neural treinada

## Estrutura do Projeto

```
.
├── app.py                 # Aplicação Flask principal
├── requirements.txt       # Dependências do Python
├── rnn.npy               # Modelo de rede neural pré-treinado
└── src/
    ├── static/           # Arquivos estáticos (CSS, JS, imagens)
    │   ├── css/
    │   │   └── style.css
    │   └── js/
    │       ├── game.js
    │       └── training.js
    └── templates/         # Templates HTML
        └── index.html
```

## Personalização

Você pode ajustar os parâmetros de treinamento da rede neural no arquivo `training.js`:
- Número de gerações
- Jogos por geração
- Tamanho da população

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para obter mais informações.
