from cli import TicTacToeCLI
from ui.gui import MainWindow
if __name__ == "__main__":
    choice = None
    while choice not in ["1", "2"]:
        choice = input("""
Escolha:
[ 1 ] Terminal
[ 2 ] GUI
""")
        if choice not in ["1", "2"]:
            print("Opção inválida! Por favor, escolha 1 ou 2.")
    
    if choice == "1":
        TicTacToeCLI().start()
    elif choice == "2":
        app = MainWindow()
        app.mainloop()