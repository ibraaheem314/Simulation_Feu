import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==== CONFIGURATION ====
DATA_PATH = "Data/resultats_tests_comparaison.csv"
IMAGE_DIR = "Images"
DATA_DIR = "Data"

# Création des dossiers si besoin
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ==== LECTURE DU DATASET ====
df = pd.read_csv(DATA_PATH)

# ==== PRE-PROCESSING ====
df['temps_calcul'] = pd.to_numeric(df['temps_calcul'], errors='coerce')
df['temps_iteration'] = pd.to_numeric(df['temps_iteration'], errors='coerce')

# Nettoyage des colonnes de type string -> tuple
df['vent'] = df['vent'].apply(lambda x: tuple(map(int, x.strip("() ").split(","))))
df['start'] = df['start'].apply(lambda x: tuple(map(int, x.strip("() ").split(","))))

# ==== CALCUL DES ACCÉLÉRATIONS ====
df['accel_global'] = 0.0
df['accel_temp'] = 0.0

group_keys = ['longueur', 'discretisation', 'vent', 'start']
grouped = df.groupby(group_keys)

for keys, group in grouped:
    ref = group[group['nombre_de_processus'] == 1]
    if ref.empty:
        continue
    t_ref_total = ref['temps_iteration'].values[0]
    t_ref_calc = ref['temps_calcul'].values[0]

    mask = (
        (df['longueur'] == keys[0]) &
        (df['discretisation'] == keys[1]) &
        (df['vent'] == keys[2]) &
        (df['start'] == keys[3])
    )

    df.loc[mask, 'accel_global'] = t_ref_total / df.loc[mask, 'temps_iteration']
    df.loc[mask, 'accel_temp'] = df.loc[mask].apply(
        lambda row: t_ref_calc / row['temps_calcul'] if row['temps_calcul'] != 0 else 0,
        axis=1
    )

# ==== GRAPHIQUES ====
sns.set(style="whitegrid")

def plot_graph(data, x, y, title, ylabel, filename, hue='discretisation'):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x=x, y=y, hue=hue, marker='o', ci='sd')
    plt.title(title)
    plt.xlabel("Nombre de processus")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, filename))
    plt.close()

df_plot = df.drop_duplicates(subset=['longueur', 'discretisation', 'vent', 'start', 'nombre_de_processus'])

plot_graph(df_plot, "nombre_de_processus", "accel_global", "Accélération Globale", "Accélération", "accel_global.png")
plot_graph(df_plot, "nombre_de_processus", "temps_iteration", "Temps Total d'Exécution", "Temps (s)", "temps_total.png")
plot_graph(df_plot, "nombre_de_processus", "temps_calcul", "Temps Moyen de Calcul", "Temps de calcul (s)", "temps_calcul.png")

# ==== STATISTIQUES SÉQUENTIELLES ====
df_seq = df[df["nombre_de_processus"] == 1]

temps_calcul_moyen = df_seq["temps_calcul"].replace(0, np.nan).mean()
temps_affichage_moyen = df_seq["temps_affichage"].replace(0, np.nan).mean()
temps_total_iteration_moyen = df_seq["temps_iteration"].mean()
temps_total_simulation = df_seq.groupby(["longueur", "discretisation", "vent", "start"])["temps_iteration"].sum().mean()

latex_table = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Temps moyen observé en mode séquentiel}}
\\begin{{tabular}}{{@{{}}cccc@{{}}}}
\\toprule
\\textbf{{Temps de calcul (s)}} & \\textbf{{Temps d'affichage (s)}} & \\textbf{{Total / itération (s)}} & \\textbf{{Total (s)}} \\\\
\\midrule
{temps_calcul_moyen:.5f} & {temps_affichage_moyen:.5f} & {temps_total_iteration_moyen:.5f} & {temps_total_simulation:.5f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
print(latex_table)

# ==== TABLEAU DE COMPARAISON POUR UN CAS FIXE ====
example_params = {
    "longueur": 1,
    "discretisation": 100,
    "vent": (2, 3),
    "start": (50, 50)
}

filtered = df[
    (df["longueur"] == example_params["longueur"]) &
    (df["discretisation"] == example_params["discretisation"]) &
    (df["vent"] == example_params["vent"]) &
    (df["start"] == example_params["start"])
].sort_values("nombre_de_processus")

t_seq_total = filtered[filtered['nombre_de_processus'] == 1]['temps_iteration'].values[0]
t_seq_calc = filtered[filtered['nombre_de_processus'] == 1]['temps_calcul'].values[0]

# Génération tableau LaTeX
latex_comparaison = """
\\begin{table}[H]
\\centering
\\caption{Comparaison des performances selon le nombre de processus (cas spécifique)}
\\begin{tabular}{@{}ccccc@{}}
\\toprule
\\textbf{Processus} & \\textbf{Calcul (s)} & \\textbf{Total (s)} & \\textbf{Accél. globale} & \\textbf{Accél. calcul} \\\\
\\midrule
"""

for _, row in filtered.iterrows():
    p = int(row['nombre_de_processus'])
    t_calc = row['temps_calcul']
    t_total = row['temps_iteration']

    accel_globale = round(t_seq_total / t_total, 2) if t_total > 0 else "--"
    accel_calc = round(t_seq_calc / t_calc, 2) if t_calc > 0 else "--"

    latex_comparaison += f"{p} & {t_calc:.5f} & {t_total:.2f} & {accel_globale} & {accel_calc} \\\\\n"

latex_comparaison += "\\bottomrule\n\\end{tabular}\n\\end{table}"

print(latex_comparaison)
