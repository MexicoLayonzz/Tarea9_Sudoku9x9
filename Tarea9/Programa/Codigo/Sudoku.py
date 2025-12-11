import random
import copy
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# -----------------------
# Configura tu puzzle aquí (0 = vacío)
# -----------------------
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# -----------------------
# Preparación: máscara de fijos
# -----------------------
fixed = [[cell != 0 for cell in row] for row in puzzle]

# -----------------------
# Funciones utilitarias
# -----------------------
def initial_row_fill(row_idx):
    fixed_vals = [puzzle[row_idx][j] for j in range(9) if fixed[row_idx][j]]
    missing = [n for n in range(1,10) if n not in fixed_vals]
    row = puzzle[row_idx][:]
    pos = [j for j in range(9) if not fixed[row_idx][j]]
    random.shuffle(missing)
    for j, val in zip(pos, missing):
        row[j] = val
    return row

def make_individual():
    return [initial_row_fill(i) for i in range(9)]

def fitness(ind):
    score = 0
    # columnas
    for c in range(9):
        col = [ind[r][c] for r in range(9)]
        score += len(set(col))
    # bloques 3x3
    for br in range(3):
        for bc in range(3):
            cells = []
            for r in range(br*3, br*3+3):
                for c in range(bc*3, bc*3+3):
                    cells.append(ind[r][c])
            score += len(set(cells))
    return score  # máximo 162

def row_crossover(row_a, row_b, row_idx):
    # escoger base de uno de los padres y reparar
    base = row_a[:] if random.random() < 0.5 else row_b[:]
    result = base[:]
    # asegurar valores fijos
    for j in range(9):
        if fixed[row_idx][j]:
            result[j] = puzzle[row_idx][j]
    # reparar duplicados / ceros
    present = set(result)
    missing = [n for n in range(1,10) if n not in present]
    counts = {}
    for j in range(9):
        v = result[j]
        counts[v] = counts.get(v, 0) + 1
    to_replace = []
    for j in range(9):
        if not fixed[row_idx][j]:
            if counts[result[j]] > 1 or result[j] == 0:
                to_replace.append(j)
                counts[result[j]] -= 1
    random.shuffle(missing)
    for j, val in zip(to_replace, missing):
        result[j] = val
    return result

def crossover(parent_a, parent_b):
    child = []
    for r in range(9):
        child.append(row_crossover(parent_a[r], parent_b[r], r))
    return child

def mutate(individual, mutation_rate=0.06):
    child = copy.deepcopy(individual)
    for r in range(9):
        if random.random() < mutation_rate:
            non_fixed_indices = [j for j in range(9) if not fixed[r][j]]
            if len(non_fixed_indices) >= 2:
                a, b = random.sample(non_fixed_indices, 2)
                child[r][a], child[r][b] = child[r][b], child[r][a]
    return child

def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(list(range(len(pop))), k)
    selected.sort(key=lambda i: fitnesses[i], reverse=True)
    return pop[selected[0]]

# -----------------------
# Algoritmo Genético (elitista)
# -----------------------
def run_ga(pop_size=500, generations=2000, mutation_rate=0.06, elites=5, verbose=True):
    pop = [make_individual() for _ in range(pop_size)]
    best_history = []
    for gen in range(1, generations+1):
        fitnesses = [fitness(ind) for ind in pop]
        sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)
        pop = [pop[i] for i in sorted_idx]
        fitnesses = [fitnesses[i] for i in sorted_idx]
        best = pop[0]
        best_f = fitnesses[0]
        best_history.append(best_f)
        if gen % 50 == 0 or gen == 1:
            if verbose:
                print(f"Gen {gen:4d}  Best fitness: {best_f}/162")
        if best_f == 162:
            if verbose:
                print(f"SOLUTION found at generation {gen}!")
            return best, best_history, gen
        # elitismo
        next_pop = [copy.deepcopy(pop[i]) for i in range(elites)]
        while len(next_pop) < pop_size:
            parent_a = tournament_selection(pop, fitnesses, k=3)
            parent_b = tournament_selection(pop, fitnesses, k=3)
            child = crossover(parent_a, parent_b)
            child = mutate(child, mutation_rate)
            next_pop.append(child)
        pop = next_pop
    # si no se encontró solución, devolver mejor encontrado
    fitnesses = [fitness(ind) for ind in pop]
    best_idx = int(np.argmax(fitnesses))
    return pop[best_idx], best_history, generations

# -----------------------
# Ejecutar GA
# -----------------------
# -----------------------
# Ejecutar GA
# -----------------------
class SudokuGAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver - Algoritmo Genético")
        self.root.geometry("1200x700")
        self.running = False
        self.best_solution = None
        self.history = []
        
        # Panel superior con controles
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Iniciar", command=self.start_ga).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Detener", command=self.stop_ga).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Verificar", command=self.verify_solution).pack(side=tk.LEFT, padx=5)
        # Control para que el usuario elija cuántas generaciones ejecutar
        ttk.Label(control_frame, text="Generaciones:").pack(side=tk.LEFT, padx=(10,2))
        self.generations_var = tk.IntVar(value=5000)
        # Spinbox de Tk (width pequeño)
        self.gen_spin = tk.Spinbox(control_frame, from_=1, to=20000, increment=100, textvariable=self.generations_var, width=8)
        self.gen_spin.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Estado: Listo", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Panel principal con dos columnas
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Columna izquierda: Gráfico de fitness
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_frame, text="Evolución del Fitness", font=("Arial", 12, "bold")).pack()
        self.fig = Figure(figsize=(6, 5), dpi=80)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Columna derecha: Información y solución
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        ttk.Label(right_frame, text="Información", font=("Arial", 12, "bold")).pack()
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.gen_label = ttk.Label(info_frame, text="Generación: 0", font=("Arial", 10))
        self.gen_label.pack(anchor=tk.W)
        
        self.fitness_label = ttk.Label(info_frame, text="Fitness: 0/162", font=("Arial", 10))
        self.fitness_label.pack(anchor=tk.W)
        
        self.time_label = ttk.Label(info_frame, text="Tiempo: 0s", font=("Arial", 10))
        self.time_label.pack(anchor=tk.W)
        
        ttk.Label(right_frame, text="Mejor Solución Encontrada", font=("Arial", 12, "bold")).pack(pady=(10, 0))
        self.sudoku_text = tk.Text(right_frame, height=12, width=25, font=("Courier", 9))
        self.sudoku_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def start_ga(self):
        if not self.running:
            self.running = True
            self.history = []
            self.status_label.config(text="Estado: En ejecución...")
            thread = threading.Thread(target=self.run_ga_thread, daemon=True)
            thread.start()
    
    def stop_ga(self):
        self.running = False
        self.status_label.config(text="Estado: Detenido")
    
    def run_ga_thread(self):
        import time
        start_time = time.time()
        # Leer el número de generaciones desde el control de la GUI
        try:
            gens_to_run = int(self.generations_var.get())
        except Exception:
            gens_to_run = 5000
        best, history, gens = run_ga_with_callback(
            pop_size=400, 
            generations=gens_to_run, 
            mutation_rate=0.08, 
            elites=10,
            callback=self.update_ui,
            stop_flag=lambda: not self.running
        )
        
        self.best_solution = best
        self.history = history
        elapsed = time.time() - start_time
        
        self.status_label.config(text=f"Estado: Completado en {elapsed:.1f}s - Generaciones: {gens}")
        self.display_solution(best)
        self.update_graph()
        # Ejecutar verificación automática al completarse la ejecución
        try:
            self.verify_solution()
        except Exception:
            # no fallar la interfaz por la verificación
            pass
        self.running = False
    
    def update_ui(self, gen, best_f, elapsed):
        self.root.after(0, self._update_ui_main, gen, best_f, elapsed)
    
    def _update_ui_main(self, gen, best_f, elapsed):
        self.gen_label.config(text=f"Generación: {gen}")
        self.fitness_label.config(text=f"Fitness: {best_f}/162")
        self.time_label.config(text=f"Tiempo: {elapsed:.1f}s")
        self.update_graph()
    
    def update_graph(self):
        if self.history:
            self.ax.clear()
            self.ax.plot(self.history, linewidth=2, color='#2E86AB')
            self.ax.axhline(y=162, color='green', linestyle='--', label='Objetivo (162)')
            self.ax.set_xlabel("Generación")
            self.ax.set_ylabel("Mejor Fitness")
            self.ax.set_title("Evolución del Fitness")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw_idle()
    
    def display_solution(self, solution):
        self.sudoku_text.config(state=tk.NORMAL)
        self.sudoku_text.delete(1.0, tk.END)
        
        text = ""
        for r in range(9):
            if r % 3 == 0:
                text += "+-------+-------+-------+\n"
            text += "| "
            for c in range(9):
                text += str(solution[r][c]) + " "
                if (c + 1) % 3 == 0:
                    text += "| "
            text += "\n"
        text += "+-------+-------+-------+\n"
        
        self.sudoku_text.insert(tk.END, text)
        self.sudoku_text.config(state=tk.DISABLED)

    def verify_solution(self):
        """Verifica la mejor solución encontrada (self.best_solution) y muestra conflictos."""
        if not self.best_solution:
            messagebox.showwarning("Verificar", "No hay solución disponible. Ejecuta el algoritmo primero.")
            return
        cols, blocks = find_conflicts(self.best_solution)
        valid = is_valid(self.best_solution)
        msg_lines = [f"Válido completo?: {valid}"]
        if not cols and not blocks:
            msg_lines.append("No se detectaron conflictos.")
        else:
            if cols:
                msg_lines.append("Conflictos en columnas:")
                for c, counts in cols:
                    parts = [f"{v}->filas{rows}" for v, rows in counts.items()]
                    msg_lines.append(f"  Col {c}: " + ", ".join(parts))
            if blocks:
                msg_lines.append("Conflictos en bloques:")
                for (br, bc), counts in blocks:
                    parts = [f"{v}->pos{poses}" for v, poses in counts.items()]
                    msg_lines.append(f"  Bloque ({br},{bc}): " + ", ".join(parts))
        msg = "\n".join(msg_lines)
        messagebox.showinfo("Verificar", msg)
        print(msg)

def run_ga_with_callback(pop_size=500, generations=2000, mutation_rate=0.06, elites=5, callback=None, stop_flag=None):
    import time
    start_time = time.time()
    pop = [make_individual() for _ in range(pop_size)]
    best_history = []
    
    for gen in range(1, generations+1):
        if stop_flag and stop_flag():
            break
            
        fitnesses = [fitness(ind) for ind in pop]
        sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i], reverse=True)
        pop = [pop[i] for i in sorted_idx]
        fitnesses = [fitnesses[i] for i in sorted_idx]
        best = pop[0]
        best_f = fitnesses[0]
        best_history.append(best_f)
        
        elapsed = time.time() - start_time
        if callback:
            callback(gen, best_f, elapsed)
        
        if best_f == 162:
            print(f"SOLUTION found at generation {gen}!")
            return best, best_history, gen
        
        # elitismo
        next_pop = [copy.deepcopy(pop[i]) for i in range(elites)]
        while len(next_pop) < pop_size:
            parent_a = tournament_selection(pop, fitnesses, k=3)
            parent_b = tournament_selection(pop, fitnesses, k=3)
            child = crossover(parent_a, parent_b)
            child = mutate(child, mutation_rate)
            next_pop.append(child)
        pop = next_pop
    
    # si no se encontró solución, devolver mejor encontrado
    fitnesses = [fitness(ind) for ind in pop]
    best_idx = int(np.argmax(fitnesses))
    return pop[best_idx], best_history, generations

def find_conflicts(solution):
    col_conflicts = []
    for c in range(9):
        col = [solution[r][c] for r in range(9)]
        if len(set(col)) != 9:
            counts = {}
            for r, v in enumerate(col):
                counts.setdefault(v, []).append(r)
            col_conflicts.append((c, counts))  # (col_index, {value:[rows...]})
    block_conflicts = []
    for br in range(3):
        for bc in range(3):
            cells = []
            positions = []
            for r in range(br*3, br*3+3):
                for c in range(bc*3, bc*3+3):
                    cells.append(solution[r][c])
                    positions.append((r, c))
            if len(set(cells)) != 9:
                counts = {}
                for pos, val in zip(positions, cells):
                    counts.setdefault(val, []).append(pos)
                block_conflicts.append(((br, bc), counts))  # ((block_row,block_col), {value:[(r,c),...]})
    return col_conflicts, block_conflicts

def is_valid(solution):
    # Verifica columnas y bloques (las filas deberían estar completas por construcción)
    for c in range(9):
        if set(solution[r][c] for r in range(9)) != set(range(1,10)):
            return False
    for br in range(3):
        for bc in range(3):
            vals = [solution[r][c] for r in range(br*3, br*3+3) for c in range(bc*3, bc*3+3)]
            if set(vals) != set(range(1,10)):
                return False
    return True

# (Opcional) Para verificar desde REPL o tras ejecutar la GUI:
# por ejemplo, después de cerrar la ventana GUI puedes usar:
# cols, blocks = find_conflicts(app.best_solution)
# print("Válido completo?:", is_valid(app.best_solution))
# print("Conflictos en columnas (index -> value:rows):", cols)
# print("Conflictos en bloques ((br,bc) -> value:positions):", blocks)

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    root = tk.Tk()
    app = SudokuGAApp(root)
    root.mainloop()
