from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from usecases.genetic_algorithm import GeneticAlgorithm
from adapters.minimax_trainer import MinimaxTrainer
from entities.neural_network import NeuralNetwork
from adapters.minimax_player import MinimaxPlayer
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from utils.utils import WIN_LINES
import tkinter as tk
import numpy as np
import threading
import hashlib
import time
import os

class FrameJogarVsMinimax(tk.Frame):
    def __init__(self, master, voltar_callback):
        super().__init__(master)
        self.master = master
        self.voltar_callback = voltar_callback
        self.board = np.zeros((3, 3), dtype=int)
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.minimax = None
        self.dificuldade = None
        self.dificuldade_frame = None
        self.turno_var = tk.StringVar(value="")
        self.turno_label = tk.Label(self, textvariable=self.turno_var, font=("Arial", 12, "bold"), fg="#333")
        self.turno_label.pack(pady=4)
        self.create_widgets()
        self.create_dificuldade_buttons()

    def create_widgets(self):
        self.turno_var.set("Turno: Humano (O)")
        self.board_frame = tk.Frame(self)
        self.board_frame.pack(pady=20)
        for i in range(3):
            for j in range(3):
                btn = tk.Button(self.board_frame, text="", width=6, height=3, font=("Arial", 18),
                                command=lambda r=i, c=j: self.on_click(r, c))
                btn.grid(row=i, column=j, padx=2, pady=2)
                self.buttons[i][j] = btn
        self.btn_reiniciar = tk.Button(self, text="Reiniciar", command=self.reset_board)
        self.btn_voltar = tk.Button(self, text="Voltar", command=self.voltar_callback)

    def create_dificuldade_buttons(self):
        self.dificuldade_frame = tk.Frame(self)
        self.dificuldade_frame.pack(pady=10)
        tk.Label(self.dificuldade_frame, text="Escolha a dificuldade:").pack(pady=2)
        tk.Button(self.dificuldade_frame, text="Easy", width=10, command=lambda: self.set_dificuldade("easy")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.dificuldade_frame, text="Medium", width=10, command=lambda: self.set_dificuldade("medium")).pack(side=tk.LEFT, padx=5)
        tk.Button(self.dificuldade_frame, text="Hard", width=10, command=lambda: self.set_dificuldade("hard")).pack(side=tk.LEFT, padx=5)

    def set_dificuldade(self, op):
        self.dificuldade = op
        if op == "easy":
            self.minimax = MinimaxTrainer(p_minimax=0.2)
        elif op == "medium":
            self.minimax = MinimaxTrainer(p_minimax=0.5)
        else:
            self.minimax = MinimaxPlayer()
        if self.dificuldade_frame:
            self.dificuldade_frame.destroy()
        self.btn_reiniciar.pack(pady=5)
        self.btn_voltar.pack(pady=5)
        self.reset_board()
        self.turno_var.set("Turno: Humano (O)")

    def on_click(self, r, c):
        if not self.minimax:
            return
        if self.board[r, c] != 0:
            messagebox.showwarning("Aviso", "Posição já ocupada!")
            return
        self.board[r, c] = -1  # Jogador humano é O (-1)
        self.update_buttons()
        winner = self.check_winner()
        if winner is not None:
            self.show_result(winner)
            return
        # Jogada do Minimax (X)
        if isinstance(self.minimax, MinimaxTrainer):
            import random
            modo = "Aleatório"
            if random.random() <= self.minimax.p_minimax:
                modo = "Minimax"
            self.turno_var.set(f"Turno: {modo} (X)")
        else:
            self.turno_var.set("Turno: Minimax (X)")
        row, col = self.minimax.move(self.board.tolist())
        if row == -1 or self.board[row, col] != 0:
            winner = self.check_winner()
            self.show_result(winner)
            return
        self.board[row, col] = +1
        self.update_buttons()
        winner = self.check_winner()
        if winner is not None:
            self.show_result(winner)
        self.turno_var.set("Turno: Humano (O)")

    def update_buttons(self):
        for i in range(3):
            for j in range(3):
                v = self.board[i, j]
                if v == +1:
                    self.buttons[i][j]["text"] = "X"
                    self.buttons[i][j]["fg"] = "blue"
                elif v == -1:
                    self.buttons[i][j]["text"] = "O"
                    self.buttons[i][j]["fg"] = "red"
                else:
                    self.buttons[i][j]["text"] = ""
                    self.buttons[i][j]["fg"] = "black"

    def check_winner(self):
        for line in WIN_LINES:
            s = sum(self.board[r, c] for r, c in line)
            if s == +3:
                return +1  # Minimax vence
            if s == -3:
                return -1  # Humano vence
        if not (self.board == 0).any():
            return 0  # Empate
        return None

    def show_result(self, winner):
        if winner == +1:
            msg = "Minimax (X) venceu!"
        elif winner == -1:
            msg = "Você (O) venceu!"
        else:
            msg = "Empate!"
        messagebox.showinfo("Fim de Jogo", msg)

    def reset_board(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.update_buttons()
        self.turno_var.set("Turno: Humano (O)")

class FrameTreinarRede(tk.Frame):
    def __init__(self, master, voltar_callback, jogar_callback):
        super().__init__(master)
        self.master = master
        self.voltar_callback = voltar_callback
        self.jogar_callback = jogar_callback
        self.progress = None
        self.fitness_history = []
        self.running = False
        self.progress_ready = False
        self.info_var = tk.StringVar(value="")
        self.all_fitness = []
        self.mean_fitness = []
        self.create_widgets()

    def create_widgets(self):
        form = tk.Frame(self)
        form.pack(pady=10)
        tk.Label(form, text="População:").grid(row=0, column=0, sticky="e")
        self.pop_var = tk.IntVar(value=20)
        tk.Entry(form, textvariable=self.pop_var, width=8).grid(row=0, column=1)
        tk.Label(form, text="Gerações:").grid(row=1, column=0, sticky="e")
        self.gen_var = tk.IntVar(value=10)
        tk.Entry(form, textvariable=self.gen_var, width=8).grid(row=1, column=1)
        tk.Label(form, text="Partidas/Indivíduo:").grid(row=2, column=0, sticky="e")
        self.games_var = tk.IntVar(value=10)
        tk.Entry(form, textvariable=self.games_var, width=8).grid(row=2, column=1)
        self.progress = ttk.Progressbar(self, orient="horizontal", length=250, mode="determinate")
        self.progress.pack(pady=5)
        self.progress_ready = True
        self.btn_iniciar = tk.Button(self, text="Iniciar", command=self.iniciar_treino)
        self.btn_iniciar.pack(pady=10)
        self.status_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.status_var).pack(pady=2)
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(pady=10)
        self.info_label = tk.Label(self, textvariable=self.info_var, justify="left", anchor="w", font=("Arial", 11), wraplength=380)
        self.info_label.pack(pady=8, fill="x")
        self.test_frame = tk.Frame(self)
        self.test_frame.pack(pady=5)
        tk.Label(self.test_frame, text="Testar acurácia contra:").pack(side=tk.LEFT, padx=2)
        self.test_opcao = tk.StringVar(value="difícil")
        self.btn_testar_dificil = tk.Button(self.test_frame, text="Minimax Difícil", command=lambda: self.testar_acuracia('difícil'), state="disabled", width=16)
        self.btn_testar_dificil.pack(side=tk.LEFT, padx=2)
        self.btn_testar_medio = tk.Button(self.test_frame, text="Minimax Médio", command=lambda: self.testar_acuracia('médio'), state="disabled", width=16)
        self.btn_testar_medio.pack(side=tk.LEFT, padx=2)
        self.test_result_var = tk.StringVar(value="")
        self.test_result_label = tk.Label(
            self,
            textvariable=self.test_result_var,
            justify="left",
            anchor="w",
            font=("Arial", 12, "bold"),
            wraplength=380,
            fg="#003366",
            bg="white",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6
        )
        self.test_result_label.pack(pady=8, fill="x")
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(pady=10)
        self.btn_jogar = tk.Button(self.buttons_frame, text="Jogar contra Rede Treinada", command=self.jogar_callback, state="disabled", width=28)
        self.btn_jogar.pack(side=tk.LEFT, padx=10)
        self.btn_voltar = tk.Button(self.buttons_frame, text="Voltar", command=self.voltar_callback, width=10)
        self.btn_voltar.pack(side=tk.LEFT, padx=10)
        self.all_fitness = []  # fitness de todos os indivíduos por geração
        self.mean_fitness = []

    def iniciar_treino(self):
        if self.running or not self.progress_ready or self.progress is None:
            messagebox.showwarning("Aviso", "Aguarde a interface carregar antes de iniciar o treino.")
            return
        self.running = True
        self.btn_iniciar.config(state="disabled")
        self.status_var.set("Treinando...")
        self.fitness_history = []
        self.all_fitness = []
        self.mean_fitness = []
        self.progress["value"] = 0
        self.clear_canvas()
        threading.Thread(target=self.run_treino, daemon=True).start()

    def run_treino(self):
        try:
            pop = self.pop_var.get()
            gens = self.gen_var.get()
            games = self.games_var.get()
            ga = GeneticAlgorithm(population_size=pop, generations=gens, n_games=games)
            best_fit = -float("inf")
            best_weights = None
            best_gen = 1
            start_time = time.time()
            self.all_fitness = []
            self.mean_fitness = []
            for g in range(1, gens + 1):
                pop_list = ga._init_pop() if g == 1 else pop_list
                for c in pop_list:
                    c.score = ga.evaluator.evaluate(c.weights_vector)
                pop_list.sort(key=lambda c: c.score, reverse=True)
                self.fitness_history.append(pop_list[0].score)
                self.all_fitness.append([c.score for c in pop_list])
                self.mean_fitness.append(sum(c.score for c in pop_list) / len(pop_list))
                if pop_list[0].score > best_fit:
                    best_fit = pop_list[0].score
                    best_weights = pop_list[0].weights_vector.copy()
                    best_gen = g
                ga._save_population_csv(g, pop_list)
                best = pop_list[0]
                next_pop = [best]
                while len(next_pop) < pop:
                    p1 = ga._select_tournament(pop_list)
                    p2 = ga._select_tournament(pop_list)
                    while p2 is p1:
                        p2 = ga._select_tournament(pop_list)
                    child = ga._crossover(p1, p2)
                    if g > round(gens * 0.3):
                        ga._mutate(child)
                    next_pop.append(child)
                pop_list = next_pop
                self.update_progress(g, gens)
            np.save("best_network.npy", best_weights)
            elapsed = time.time() - start_time
            self.status_var.set("Treinamento concluído! Melhor rede salva em best_network.npy")
            self.plot_fitness()
            info = (
                f"\nResumo do Treinamento:\n"
                f"- Fitness final: {best_fit:.2f}\n"
                f"- Geração do melhor indivíduo: {best_gen}\n"
                f"- Shape dos pesos: {best_weights.shape}\n"
                f"- Tempo total: {elapsed:.1f} segundos\n"
                f"- Parâmetros: População={pop}, Gerações={gens}, Partidas/Indivíduo={games}\n"
                f"\nArquivo salvo: best_network.npy"
            )
            self.info_var.set(info)
            self.btn_jogar.config(state="normal")
            self.btn_testar_dificil.config(state="normal")
            self.btn_testar_medio.config(state="normal")
        except Exception as e:
            self.status_var.set(f"Erro: {e}")
            messagebox.showerror("Erro", str(e))
        finally:
            self.btn_iniciar.config(state="normal")
            self.running = False

    def update_progress(self, g, total):
        if self.progress is not None:
            self.progress["value"] = int(100 * g / total)
            self.progress.update()

    def plot_fitness(self):
        self.clear_canvas()
        if not self.fitness_history:
            return
        fig = Figure(figsize=(3.5,2.5), dpi=100)
        ax = fig.add_subplot(111)
        # Scatter de todos os fitness (ruim)
        for g, fit_list in enumerate(self.all_fitness):
            ax.scatter([g+1]*len(fit_list), fit_list, color="#d62728", s=12, alpha=0.7, label="Indivíduos" if g==0 else "")
        # Linha do melhor (verde)
        ax.plot(range(1, len(self.fitness_history)+1), self.fitness_history, marker="o", color="#1ca81c", label="Melhor")
        # Linha do fitness médio (amarelo)
        if self.mean_fitness:
            ax.plot(range(1, len(self.mean_fitness)+1), self.mean_fitness, marker="s", color="#e6b800", linestyle="--", label="Média")
        ax.set_title("Fitness por Geração")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Fitness")
        ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def testar_acuracia(self, modo):
        try:
            weights = np.load("best_network.npy")
            nn = NeuralNetwork(9, 9, 9, weights)
            if modo == 'difícil':
                from adapters.minimax_player import MinimaxPlayer
                adversario = MinimaxPlayer()
                adversario_nome = "Minimax Difícil"
            else:
                from adapters.minimax_trainer import MinimaxTrainer
                adversario = MinimaxTrainer(p_minimax=0.5)
                adversario_nome = "Minimax Médio"
            total = 100
            vitorias = empates = derrotas = 0
            for _ in range(total):
                board = np.zeros((3,3), dtype=int)
                turn = +1  # Rede começa
                while True:
                    if turn == +1:
                        idx = nn.predict(board.flatten())
                        r, c = divmod(idx, 3)
                    else:
                        r, c = adversario.move(board.tolist())
                    if board[r, c] != 0:
                        derrotas += 1
                        break
                    board[r, c] = turn
                    winner = None
                    for line in WIN_LINES:
                        s = sum(board[r, c] for r, c in line)
                        if s == +3:
                            winner = +1
                        elif s == -3:
                            winner = -1
                    if winner is not None:
                        if winner == +1:
                            vitorias += 1
                        else:
                            derrotas += 1
                        break
                    if not (board == 0).any():
                        empates += 1
                        break
                    turn *= -1
            taxa_vit = vitorias/total*100
            taxa_emp = empates/total*100
            taxa_der = derrotas/total*100
            import hashlib
            sha = hashlib.sha256(weights.tobytes()).hexdigest()[:12]
            preview = ", ".join(f"{v:.3f}" for v in weights[:5])
            resumo = (
                f"\nAcurácia da rede contra {adversario_nome} ({total} jogos):\n"
                f"- Vitórias: {vitorias} ({taxa_vit:.1f}%)\n"
                f"- Empates: {empates} ({taxa_emp:.1f}%)\n"
                f"- Derrotas: {derrotas} ({taxa_der:.1f}%)\n"
                f"- Shape dos pesos: {weights.shape}\n"
                f"- Primeiros valores: {preview}...\n"
                f"- Hash SHA256: {sha}\n"
            )
            self.test_result_var.set(resumo)
        except Exception as e:
            self.test_result_var.set("")
            messagebox.showerror("Erro", f"Erro ao testar acurácia: {e}")

    def clear_canvas(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

    def show(self):
        self.pack(expand=True)
        if hasattr(self, 'btn_testar_dificil') and self.btn_testar_dificil:
            self.btn_testar_dificil.config(state="normal" if os.path.exists("best_network.npy") else "disabled")
            self.btn_testar_medio.config(state="normal" if os.path.exists("best_network.npy") else "disabled")

class FrameJogarVsRedeTreinada(tk.Frame):
    def __init__(self, master, voltar_callback):
        super().__init__(master)
        self.master = master
        self.voltar_callback = voltar_callback
        self.board = np.zeros((3, 3), dtype=int)
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.nn = None
        self.humano_comeca = True
        self.turno_var = tk.StringVar(value="")
        self.turno_label = tk.Label(self, textvariable=self.turno_var, font=("Arial", 12, "bold"), fg="#333")
        self.turno_label.pack(pady=4)
        self.info_var = tk.StringVar(value="")
        self.create_widgets()
        if self.load_network():
            self.reset_board()

    def create_widgets(self):
        self.turno_var.set("Turno: Humano (O)")
        self.board_frame = tk.Frame(self)
        self.board_frame.pack(pady=20)
        for i in range(3):
            for j in range(3):
                btn = tk.Button(self.board_frame, text="", width=6, height=3, font=("Arial", 18),
                                command=lambda r=i, c=j: self.on_click(r, c))
                btn.grid(row=i, column=j, padx=2, pady=2)
                self.buttons[i][j] = btn
        self.info_label = tk.Label(self, textvariable=self.info_var, justify="left", anchor="w", font=("Arial", 10), wraplength=380)
        self.info_label.pack(pady=8, fill="x")
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(pady=10)
        self.btn_reiniciar = tk.Button(self.buttons_frame, text="Reiniciar", command=self.ask_restart, width=12)
        self.btn_reiniciar.pack(side=tk.LEFT, padx=10)
        self.btn_voltar = tk.Button(self.buttons_frame, text="Voltar", command=self.voltar_callback, width=12)
        self.btn_voltar.pack(side=tk.LEFT, padx=10)

    def ask_restart(self):
        resp = messagebox.askquestion("Reiniciar", "Quem começa?\nSim = Humano, Não = Rede IA", icon='question')
        if resp == 'yes':
            self.humano_comeca = True
        else:
            self.humano_comeca = False
        self.reset_board()

    def load_network(self):
        try:
            if not os.path.exists("best_network.npy"):
                messagebox.showerror("Erro", "Arquivo best_network.npy não encontrado!\nTreine a rede antes de jogar contra ela.")
                self.voltar_callback()
                return False
            weights = np.load("best_network.npy")
            self.nn = NeuralNetwork(9, 9, 9, weights)
            # Exibir resumo da rede carregada
            sha = hashlib.sha256(weights.tobytes()).hexdigest()[:12]
            preview = ", ".join(f"{v:.3f}" for v in weights[:5])
            info = (
                f"Rede carregada:\n"
                f"- Shape: {weights.shape}\n"
                f"- Primeiros valores: {preview}...\n"
                f"- Hash SHA256: {sha}"
            )
            self.info_var.set(info)
            return True
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar rede: {e}")
            self.voltar_callback()
            return False

    def on_click(self, r, c):
        if not self.nn:
            return
        if self.board[r, c] != 0:
            messagebox.showwarning("Aviso", "Posição já ocupada!")
            return
        self.board[r, c] = -1  # Jogador humano é O (-1)
        self.update_buttons()
        winner = self.check_winner()
        if winner is not None:
            self.show_result(winner)
            return
        self.turno_var.set("Turno: Rede Treinada (X)")
        self.rede_move()

    def rede_move(self):
        if not self.nn:
            return
        idx = self.nn.predict(self.board.flatten())
        if idx == -1:
            winner = self.check_winner()
            self.show_result(winner)
            return
        r, c = divmod(idx, 3)
        if self.board[r, c] == 0:
            self.board[r, c] = +1
            self.update_buttons()
            winner = self.check_winner()
            if winner is not None:
                self.show_result(winner)
                return
            self.turno_var.set("Turno: Humano (O)")

    def update_buttons(self):
        for i in range(3):
            for j in range(3):
                v = self.board[i, j]
                if v == +1:
                    self.buttons[i][j]["text"] = "X"
                    self.buttons[i][j]["fg"] = "blue"
                elif v == -1:
                    self.buttons[i][j]["text"] = "O"
                    self.buttons[i][j]["fg"] = "red"
                else:
                    self.buttons[i][j]["text"] = ""
                    self.buttons[i][j]["fg"] = "black"

    def check_winner(self):
        for line in WIN_LINES:
            s = sum(self.board[r, c] for r, c in line)
            if s == +3:
                return +1  # Rede vence
            if s == -3:
                return -1  # Humano vence
        if not (self.board == 0).any():
            return 0  # Empate
        return None

    def show_result(self, winner):
        if winner == +1:
            msg = "Rede Neural (X) venceu!"
        elif winner == -1:
            msg = "Você (O) venceu!"
        else:
            msg = "Empate!"
        messagebox.showinfo("Fim de Jogo", msg)

    def reset_board(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.update_buttons()
        if not self.humano_comeca and self.nn:
            self.turno_var.set("Turno: Rede Treinada (X)")
            self.rede_move()
        else:
            self.turno_var.set("Turno: Humano (O)")

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("T2IA - Jogo da Velha Inteligente")
        self.geometry("400x600")
        self.resizable(False, False)
        self.current_frame = None
        self.create_main_menu()

    def create_main_menu(self):
        if self.current_frame:
            self.current_frame.destroy()
        frame = tk.Frame(self)
        frame.pack(expand=True)
        
        btn_minimax = tk.Button(frame, text="Jogar vs Minimax", width=25, command=self.show_minimax_frame)
        btn_minimax.pack(pady=10)
        
        btn_treinar = tk.Button(frame, text="Treinar Rede", width=25, command=self.show_treinar_frame)
        btn_treinar.pack(pady=10)
        
        btn_rede = tk.Button(frame, text="Jogar vs Rede Treinada", width=25, command=self.show_rede_frame)
        btn_rede.pack(pady=10)
        
        btn_sair = tk.Button(frame, text="Sair", width=25, command=self.quit)
        btn_sair.pack(pady=10)
        
        self.current_frame = frame

    def show_minimax_frame(self):
        if self.current_frame:
            self.current_frame.destroy()
        frame = FrameJogarVsMinimax(self, self.create_main_menu)
        frame.pack(expand=True)
        self.current_frame = frame

    def show_treinar_frame(self):
        if self.current_frame:
            self.current_frame.destroy()
        frame = FrameTreinarRede(self, self.create_main_menu, self.show_rede_frame)
        frame.pack(expand=True)
        self.current_frame = frame
        if hasattr(frame, 'btn_testar_dificil') and frame.btn_testar_dificil:
            frame.btn_testar_dificil.config(state="normal" if os.path.exists("best_network.npy") else "disabled")
            frame.btn_testar_medio.config(state="normal" if os.path.exists("best_network.npy") else "disabled")

    def show_rede_frame(self):
        if self.current_frame:
            self.current_frame.destroy()
        frame = FrameJogarVsRedeTreinada(self, self.create_main_menu)
        frame.pack(expand=True)
        self.current_frame = frame