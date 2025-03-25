# Modele de propagation d'incendie
import numpy as np

ROW    = 1
COLUMN = 0

def pseudo_random ( t_index:int, t_time_step:int ) :
    """
    Calcul un nombre pseudo-aléatoire en fonction d'un index et d'un pas de temps
    Cela permet de complètement contrôler la simulation et d'obtenir exactement la
    même simulation à chaque exécution...
    """
    xi = t_index * (t_time_step+1)
    r  = (48271*xi)%2147483647
    return r/2147483646.

def log_factor( t_value: int ):
    """
    Permet d'avoir une loi en "log" pour certaines variables modifiant la loi de probabilité.
    """
    from math import log
    return log(1.+t_value)/log(256)

class Model:
    """
    Modèle de propagation de feu dans une végétation homogène.
    """
    def __init__(self, t_length : float, t_discretization:int, t_wind_vector, t_start_fire_position, t_max_wind:float = 60.):
        """
        t_length         : Longueur du domaine (carré) en km
        t_discretization : Nombre de cellules de discrétisation par direction
        t_wind_vector    : Direction et force du vent (vecteur de deux composantes)
        t_start_fire_position : Indices lexicographiques d'où démarre l'incendie
        t_max_wind       : Vitesse du vent (en km/h) à partir duquel l'incendie ne peut plus se propager dans la direction opposée à l'incendie
        """
        from math import sqrt
        from numpy import linalg

        if t_discretization <= 0:
            raise ValueError("Le nombre de cases par direction doit être plus grand que zéro.")

        self.length     = t_length
        self.geometry   = t_discretization
        self.distance   = t_length/t_discretization
        self.wind       = np.array(t_wind_vector)
        self.wind_speed = linalg.norm(self.wind)
        self.max_wind   = t_max_wind
        self.vegetation_map = 255*np.ones(shape=(t_discretization,t_discretization), dtype=np.uint8)
        self.fire_map       = np.zeros(shape=(t_discretization,t_discretization), dtype=np.uint8)
        self.fire_map[t_start_fire_position[COLUMN], t_start_fire_position[ROW]] = np.uint8(255)
        self.fire_front = { (t_start_fire_position[COLUMN], t_start_fire_position[ROW]) : np.uint8(255) }

        ALPHA0 = 4.52790762e-01
        ALPHA1 = 9.58264437e-04
        ALPHA2 = 3.61499382e-05

        self.p1 = 0.
        if self.wind_speed < self.max_wind:
            self.p1 = ALPHA0 + ALPHA1*self.wind_speed + ALPHA2*(self.wind_speed*self.wind_speed)
        else:
            self.p1 = ALPHA0 + ALPHA1*self.max_wind + ALPHA2*(self.max_wind*self.max_wind)
        self.p2 = 0.3

        if self.wind[COLUMN] > 0:
            self.alphaEastWest = abs(self.wind[COLUMN]/self.max_wind)+1
            self.alphaWestEast = 1.-abs(self.wind[COLUMN]/t_max_wind)
        else:
            self.alphaWestEast = abs(self.wind[COLUMN]/t_max_wind)+1
            self.alphaEastWest = 1. - abs(self.wind[COLUMN]/t_max_wind)

        if self.wind[ROW] > 0:
            self.alphaSouthNorth = abs(self.wind[ROW]/t_max_wind) + 1
            self.alphaNorthSouth = 1. - abs(self.wind[ROW]/self.max_wind)
        else:
            self.alphaNorthSouth = abs(self.wind[ROW]/self.max_wind) + 1
            self.alphaSouthNorth = 1. - abs(self.wind[ROW]/self.max_wind)
        self.time_step = 0

    def glob_index(self, coord ) :
        """
        Retourne un indice unique à partir des indices lexicographiques
        """
        return coord[ROW]*self.geometry + coord[COLUMN]

    def update(self) -> bool :
        """
        Mise à jour de la carte d'incendie et de végétation avec calcul de la propagation de l'incendie
        """
        import copy
        next_front = copy.deepcopy(self.fire_front)
        for lexico_coord, fire in self.fire_front.items():
            power = log_factor(fire)
            # On va tester les cases voisines pour évaluer la contamination par le feu :
            if lexico_coord[ROW] < self.geometry-1:
                tirage      = pseudo_random( self.glob_index(lexico_coord)*4059131+self.time_step, self.time_step)
                green_power = self.vegetation_map[lexico_coord[COLUMN], lexico_coord[ROW]+1] # Case au dessus
                correction  = power*log_factor(green_power)
                if tirage < self.alphaSouthNorth*self.p1*correction:
                    self.fire_map[ lexico_coord[COLUMN], lexico_coord[ROW]+1 ] = np.uint8(255)
                    next_front   [(lexico_coord[COLUMN], lexico_coord[ROW]+1)] = np.uint8(255)

            if lexico_coord[ROW] > 0:
                tirage      = pseudo_random( self.glob_index(lexico_coord)*13427+self.time_step, self.time_step)
                green_power = self.vegetation_map[lexico_coord[COLUMN], lexico_coord[ROW]-1] # Case au dessous
                correction  = power*log_factor(green_power)
                if tirage < self.alphaNorthSouth*self.p1*correction:
                    self.fire_map[ lexico_coord[COLUMN], lexico_coord[ROW]-1 ] = np.uint8(255)
                    next_front   [(lexico_coord[COLUMN], lexico_coord[ROW]-1)] = np.uint8(255)

            if lexico_coord[COLUMN] < self.geometry-1:
                tirage      = pseudo_random( self.glob_index(lexico_coord)+self.time_step*42569, self.time_step)
                green_power = self.vegetation_map[lexico_coord[COLUMN]+1, lexico_coord[ROW]]# Case à droite
                correction  = power*log_factor(green_power)
                if tirage < self.alphaEastWest*self.p1*correction:
                    self.fire_map[ lexico_coord[COLUMN]+1, lexico_coord[ROW] ] = np.uint8(255)
                    next_front   [(lexico_coord[COLUMN]+1, lexico_coord[ROW])] = np.uint8(255)

            if lexico_coord[COLUMN] > 0:
                tirage      = pseudo_random( self.glob_index(lexico_coord)*13427+self.time_step*42569, self.time_step)
                green_power = self.vegetation_map[lexico_coord[COLUMN]-1, lexico_coord[ROW]]
                correction  = power*log_factor(green_power)
                if tirage < self.alphaWestEast*self.p1*correction:
                    self.fire_map[ lexico_coord[COLUMN]-1, lexico_coord[ROW] ] = np.uint8(255)
                    next_front   [(lexico_coord[COLUMN]-1, lexico_coord[ROW])] = np.uint8(255)

            # Si le feu est à son max,
            if fire == 255:
                # On regarde si il commence à faiblir pour s'éteindre au bout d'un moment :
                tirage = pseudo_random( self.glob_index(lexico_coord) * 52513 + self.time_step, self.time_step)
                if tirage < self.p2:
                    self.fire_map[ lexico_coord[COLUMN], lexico_coord[ROW] ] >>= 1
                    next_front   [(lexico_coord[COLUMN], lexico_coord[ROW])] >>= 1
            else:
                # Foyer en train de s'éteindre.
                self.fire_map[ lexico_coord[COLUMN], lexico_coord[ROW] ] >>= 1
                next_front   [(lexico_coord[COLUMN], lexico_coord[ROW])] >>= 1
                if next_front[(lexico_coord[COLUMN], lexico_coord[ROW])] == 0:
                    next_front.pop((lexico_coord[COLUMN], lexico_coord[ROW]))
        # A chaque itération, la végétation à l'endroit d'un foyer diminue
        self.fire_front = next_front
        for lexico_coord, _ in self.fire_front.items():
            if self.vegetation_map[lexico_coord[COLUMN], lexico_coord[ROW]] > 0:
                self.vegetation_map[lexico_coord[COLUMN], lexico_coord[ROW]] -= 1
        self.time_step += 1
        return len(self.fire_front) > 0
