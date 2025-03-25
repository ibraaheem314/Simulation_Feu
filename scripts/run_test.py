import subprocess
import pandas as pd
import itertools
import time

# Définition des paramètres à tester
longueur = [1, 5, 10]  # Différentes tailles de la grille en km
discretisation = [75, 100]  # Résolution de la grille
vent = [(1, 1), (2, 3)]  # Différents vents
start = [(5, 5), (15, 10)]  # Positions initiales du feu

for N in discretisation:
    start.append((N // 2, N // 2))  # Ajouter le centre de la grille

nombre_processus = [1, 2, 4, 6, 8]  # Ajout de 1 pour le cas séquentiel

# Génération de toutes les combinaisons possibles
test_cases = list(itertools.product(longueur, discretisation, vent, start, nombre_processus))

# Liste pour stocker les résultats
resultats = []

def run_simulation(longueur, discretisation, vent, start, nombre_processus):
    """
    Exécute la simulation séquentielle ou MPI et stocke les résultats.
    """
    print(f"\n Lancement de la simulation : longueur={longueur}, discretisation={discretisation}, vent={vent}, start={start}, processus={nombre_processus}")
    
    # Déterminer la commande à exécuter
    if nombre_processus == 1:
        # Exécuter la simulation séquentielle (`simulation.py`)
        cmd = [
            "python", "simulation.py",
            "-l", str(longueur), "-n", str(discretisation), "-w", f"{vent[0]},{vent[1]}", "-s", f"{start[0]},{start[1]}"
        ]
    else:
        # Exécuter la simulation MPI (`MPI_simulation.py`)
        cmd = [
        "mpiexec", "-np", str(nombre_processus), "python", "MPI_simulation.py",
        "-l", str(longueur), "-n", str(discretisation), "-w", f"{vent[0]},{vent[1]}", "-s", f"{start[0]},{start[1]}"
        ]
    
    # Mesure du temps total de simulation
    t_start = time.perf_counter()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        t_end = time.perf_counter()
        total_time = t_end - t_start

        # Extraction des données pertinentes
        try:
            temps_calcul = float([line for line in output.split('\n') if "Temps calcul" in line][-1].split(':')[-1].strip())
            temps_affichage = float([line for line in output.split('\n') if "Temps d'affichage" in line][-1].split(':')[-1].strip())
            iteration = int([line for line in output.split('\n') if "Time step" in line][-1].split(' ')[-1].strip())
        except (IndexError, ValueError):
            print("Erreur lors de l'extraction des temps ! Vérifiez les logs.")
            return

        # Ajouter les résultats à la liste
        resultats.append({
            "iteration": iteration,
            "temps_calcul": temps_calcul,
            "temps_affichage": temps_affichage,
            "temps_iteration": total_time,
            "longueur": longueur,
            "discretisation": discretisation,
            "vent": vent,
            "start": start,
            "nombre_de_processus": nombre_processus,
            "nombre_de_coeurs": max(1, nombre_processus - 1)  # 1 cœur pour le cas séquentiel
        })
    
    except subprocess.CalledProcessError as e:
        print(f" Erreur lors de l'exécution de la simulation : {e}")
        print(e.output)

# Exécuter toutes les combinaisons
i = 1
for test in test_cases:
    print(f"\n Exécution du test {i}/{len(test_cases)}")
    longueur, discretisation, vent, start, nombre_processus = test
    run_simulation(longueur, discretisation, vent, start, nombre_processus)
    i += 1

# Sauvegarde des résultats dans un fichier CSV
df = pd.DataFrame(resultats)
df.to_csv("resultats_tests_comparaison.csv", index=False)
print("\n Données de performance sauvegardées dans 'resultats_tests_comparaison.csv' !")
