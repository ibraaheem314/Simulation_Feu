# Modele de propagation d'incendie
import numpy as np
from mpi4py import MPI
from math import log, sqrt
import copy

ROW = 1
COLUMN = 0

def decomposition(n):
	i = int(np.floor(sqrt(n)))
	while(n%i):
		i -= 1
	j = n//i
	return (max(i, j), min(i, j))

def pseudo_random(t_index: int, t_time_step: int):
	"""
	Calcul un nombre pseudo-aléatoire en fonction d'un index et d'un pas de temps
	Cela permet de complètement contrôler la simulation et d'obtenir exactement la
	même simulation à chaque exécution...
	"""
	t_index = np.int64(t_index)
	t_time_step = np.int64(t_time_step)
	xi = t_index * (t_time_step + np.int64(1))
	r = (np.int64(48271) * xi) % np.int64(2147483647)
	return r / np.int64(2147483646)



def log_factor( t_value: int ):
	"""
	Permet d'avoir une loi en "log" pour certaines variables modifiant la loi de probabilité.
	"""
	return log(1.+t_value)/log(256)

class Model:
	"""
	Modèle de propagation de feu dans une végétation homogène.
	"""
	def __init__(self, comm, rank, size, t_length : float, t_discretization:int, t_wind_vector, t_start_fire_position, t_max_wind:float = 60.):
		"""
		t_length		 : Longueur du domaine (carré) en km
		t_discretization : Nombre de cellules de discrétisation par direction
		t_wind_vector   : Direction et force du vent (vecteur de deux composantes)
		t_start_fire_position : Indices lexicographiques d'où démarre l'incendie
		t_max_wind	 : Vitesse du vent (en km/h) à partir duquel l'incendie ne peut plus se propager dans la direction opposée à l'incendie
		"""
		if t_discretization <= 0:
			raise ValueError("Le nombre de cases par direction doit être plus grand que zéro.")
		
		self.comm = comm
		self.rank = rank
		self.size = size

		self.length  = t_length
		self.geometry   = t_discretization
		self.distance   = t_length/t_discretization
		self.wind	  = np.array(t_wind_vector)
		self.wind_speed = np.linalg.norm(self.wind)
		self.max_wind   = t_max_wind
		self.vegetation_map = 255*np.ones(shape=(t_discretization,t_discretization), dtype=np.uint8)
		# self.vegetation_map = np.arange(t_discretization**2).reshape(t_discretization, t_discretization)
		self.fire_map	  = np.zeros(shape=(t_discretization,t_discretization), dtype=np.uint8)
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
		
		if self.size > 1 : 
			self.i, self.j = decomposition(self.size - 1)
			if self.i == 1:
				self.xp = t_discretization
			else : 
				self.xp = int(np.ceil(t_discretization/self.i))
			if self.j == 1:
				self.yp = t_discretization
			else : 
				self.yp = int(np.ceil(t_discretization/self.j))

			if self.rank > 0 :
				self.a = (self.rank - 1)%self.i
				self.b = (self.rank - 1)//self.i
				self.vegetation_map = self.vegetation_map[self.a*self.xp: (self.a + 1)*self.xp, self.b*self.yp:(self.b + 1)*self.yp]
				self.fire_map = self.fire_map[self.a*self.xp: (self.a + 1)*self.xp, self.b*self.yp:(self.b + 1)*self.yp]
				if not (t_start_fire_position[COLUMN] in range(self.a*self.xp, (self.a + 1)*self.xp) and t_start_fire_position[ROW] in range(self.b*self.yp, (self.b + 1)*self.yp)):
					self.fire_front = {}

	def glob_index(self, coord ) :
		"""
		Retourne un indice unique à partir des indices lexicographiques
		"""
		return coord[ROW]*self.geometry + coord[COLUMN]

	def echange_avant_update(self):
		requetes = []
		if self.b > 0 : # On envoit et reçoit du processus de gauche
			rang = self.rank - 1
			self.Sgauche = np.copy(self.vegetation_map[:, 0])
			self.Rgauche = np.empty_like(self.Sgauche)
			# print(f"Je suis le processus {self.rank} et je veux communiquer avec le processus {rang} pour lui envoyer \n{self.Sgauche}.")
			requetes.append(self.comm.Isend(self.Sgauche, dest = rang))
			requetes.append(self.comm.Irecv(self.Rgauche, source = rang))
		
		if self.b < self.j - 1 : # On envoit et reçoit du processus de droite
			rang = self.rank + 1
			self.Sdroite = np.copy(self.vegetation_map[:, -1])
			self.Rdroite = np.empty_like(self.Sdroite)
			# print(f"Je suis le processus {self.rank} et je veux communiquer avec le processus {rang} pour lui envoyer \n{self.Sdroite}.")
			requetes.append(self.comm.Isend(self.Sdroite, dest = rang))
			requetes.append(self.comm.Irecv(self.Rdroite, source = rang))
		
		if self.a > 0 : # On envoit et reçoit du processus d'en haut
			rang = self.rank - self.j
			self.Shaut = np.copy(self.vegetation_map[0, :])
			self.Rhaut = np.empty_like(self.Shaut)
			# print(f"Je suis le processus {self.rank} et je veux communiquer avec le processus {rang} pour lui envoyer \n{self.Shaut}.")
			requetes.append(self.comm.Isend(self.Shaut, dest = rang))
			requetes.append(self.comm.Irecv(self.Rhaut, source = rang))
		
		if self.a < self.i - 1 : # On envoit et reçoit du processus d'en bas
			rang = self.rank + self.j
			self.Sbas = np.copy(self.vegetation_map[-1, :])
			self.Rbas = np.empty_like(self.Sbas)
			# print(f"Je suis le processus {self.rank} et je veux communiquer avec le processus {rang} pour lui envoyer \n{self.Sbas}.")
			requetes.append(self.comm.Isend(self.Sbas, dest = rang))
			requetes.append(self.comm.Irecv(self.Rbas, source = rang))
		
		MPI.Request.Waitall(requetes)

		# if self.b > 0 : 
		# 	rang = self.rank - 1
		# 	print(f"Je suis le processus {self.rank} et après ma communication avec le processus {rang}, j'ai reçu le tableau \n{self.Rgauche}.")
		# if self.b < self.j - 1 : 
		# 	rang = self.rank + 1
		# 	print(f"Je suis le processus {self.rank} et après ma communication avec le processus {rang}, j'ai reçu le tableau \n{self.Rdroite}.")
		# if self.a > 0 : 
		# 	rang = self.rank - self.j
		# 	print(f"Je suis le processus {self.rank} et après ma communication avec le processus {rang}, j'ai reçu le tableau \n{self.Rhaut}.")
		# if self.a < self.i - 1 : 
		# 	rang = self.rank + self.j
		# 	print(f"Je suis le processus {self.rank} et après ma communication avec le processus {rang}, j'ai reçu le tableau \n{self.Rbas}.")

	def echange_pendant_update(self):
		if self.b > 0 : # On envoit et reçoit du processus de gauche
			rang = self.rank - 1
			requetes = []
			taille = len(self.listeG)
			Sgauche = np.array(self.listeG)
			requete = self.comm.isend(taille, dest = rang, tag = 0)
			requete.wait()
			requete = self.comm.irecv(source = rang, tag = 0)
			new_taille = requete.wait()
			# print(f"Je suis le processus {self.rank} et je veux envoyer {taille} coordonnées au processus {rang} qui veut lui m'envoyer {new_taille} coordonnées. (echange_pendant_update)")
			self.Rgauche = np.empty((new_taille, 2), dtype = 'i')
			requete = self.comm.isend(Sgauche, dest = rang, tag = 1)
			requete.wait()
			requete = self.comm.irecv(source = rang, tag = 1)
			self.Rgauche = requete.wait()
			# print(self.Rgauche)
			
		if self.b < self.j - 1 : # On envoit et reçoit du processus de droite
			rang = self.rank + 1
			requetes = []
			taille = len(self.listeD)
			Sdroite = np.array(self.listeD)
			requete = self.comm.isend(taille, dest = rang, tag = 0)
			requete.wait()
			requete = self.comm.irecv(source = rang, tag = 0)
			new_taille = requete.wait()
			# print(f"Je suis le processus {self.rank} et je veux envoyer {taille} coordonnées au processus {rang} qui veut lui m'envoyer {new_taille} coordonnées. (echange_pendant_update)")
			self.Rdroite = np.empty((new_taille, 2), dtype = 'i')
			requete = self.comm.isend(Sdroite, dest = rang, tag = 1)
			requete.wait()
			requete = self.comm.irecv(source = rang, tag = 1)
			self.Rdroite = requete.wait()
			# print(self.Rdroite)
			
		if self.a > 0 : # On envoit et reçoit du processus d'en haut
			rang = self.rank - self.j
			requetes = []
			taille = len(self.listeH)
			Shaut = np.array(self.listeH)
			requete = self.comm.isend(taille, dest = rang, tag = 0)
			requete.wait()
			requete = self.comm.irecv(source = rang, tag = 0)
			new_taille = requete.wait()
			# print(f"Je suis le processus {self.rank} et je veux envoyer {taille} coordonnées au processus {rang} qui veut lui m'envoyer {new_taille} coordonnées. (echange_pendant_update)")
			self.Rhaut = np.empty((new_taille, 2), dtype = 'i')
			requete = self.comm.isend(Shaut, dest = rang, tag = 1)
			requete.wait()
			requete = self.comm.irecv(source = rang, tag = 1)
			self.Rhaut = requete.wait()
			# print(self.Rhaut)
			
		if self.a < self.i - 1 : # On envoit et reçoit du processus d'en bas
			rang = self.rank + self.j
			requetes = []
			taille = len(self.listeB)
			Sbas = np.array(self.listeB)
			requete = self.comm.isend(taille, dest = rang, tag = 0)
			requete.wait()
			requete = self.comm.irecv(source = rang, tag = 0)
			new_taille = requete.wait()
			# print(f"Je suis le processus {self.rank} et je veux envoyer {taille} coordonnées au processus {rang} qui veut lui m'envoyer {new_taille} coordonnées. (echange_pendant_update)")
			self.Rbas = np.empty((new_taille, 2), dtype = 'i')
			requete = self.comm.isend(Sbas, dest = rang, tag = 1)
			requete.wait()
			requete = self.comm.irecv(source = rang, tag = 1)
			self.Rbas = requete.wait()
			# print(self.Rbas)
			
		# if self.b > 0 : 
		# 	rang = self.rank - 1
		# 	print(f"Je suis le processus {self.rank} et après ma communication avec le processus {rang}, j'ai reçu le tableau \n{self.Rgauche}.(echange_pendant_update)")
		# if self.b < self.j - 1 : 
		# 	rang = self.rank + 1
		# 	print(f"Je suis le processus {self.rank} et après ma communication avec le processus {rang}, j'ai reçu le tableau \n{self.Rdroite}.(echange_pendant_update)")
		# if self.a > 0 : 
		# 	rang = self.rank - self.j
		# 	print(f"Je suis le processus {self.rank} et après ma communication avec le processus {rang}, j'ai reçu le tableau \n{self.Rhaut}.(echange_pendant_update)")
		# if self.a < self.i - 1 : 
		# 	rang = self.rank + self.j
		# 	print(f"Je suis le processus {self.rank} et après ma communication avec le processus {rang}, j'ai reçu le tableau \n{self.Rbas}.(echange_pendant_update)")
		
	def echange_apres_update(self):
		if self.rank == 0:
			map_f = []
			map_v = []
			map_f = self.comm.gather(map_f, root = 0)
			map_v = self.comm.gather(map_v, root = 0)
			for k in range(1, self.size):
				a = (k - 1)%self.i
				b = (k - 1)//self.i
				self.vegetation_map[a*self.xp:(a + 1)*self.xp, b*self.yp:(b + 1)*self.yp] = map_v[k]
				self.fire_map[a*self.xp:(a + 1)*self.xp, b*self.yp:(b + 1)*self.yp] = map_f[k]
			# print(self.vegetation_map)
			# print(self.fire_map)
				
		else:
			map_f = self.comm.gather(np.copy(self.fire_map), root = 0)
			map_v = self.comm.gather(np.copy(self.vegetation_map), root = 0)
		
	def update(self) -> bool :
		"""
		Mise à jour de la carte d'incendie et de végétation avec calcul de la propagation de l'incendie
		"""
		if self.rank > 0 : # On s'occupe des calculs
			# On récupère les données périphériques chez les autres processus
			self.echange_avant_update()

			# On initialize les listes à envoyer aux autres processus
			if self.b > 0:
				self.listeG = []
			if self.b < self.j - 1:
				self.listeD = []
			if self.a > 0:
				self.listeH = []
			if self.a < self.i - 1:
				self.listeB = []
			
			next_front = copy.deepcopy(self.fire_front)
			for lexico_coord, fire in self.fire_front.items():
				power = log_factor(fire)
				C = lexico_coord[COLUMN]
				R = lexico_coord[ROW]
				colonne = C - self.a*self.xp
				ligne = R - self.b*self.yp
				
				# On va tester les cases voisines pour évaluer la contamination par le feu :
				if R < self.geometry-1: 
					tirage	= pseudo_random( self.glob_index(lexico_coord)*4059131+self.time_step, self.time_step)
					if ligne == self.yp - 1: # Ca implique que self.b < self.j - 1
						green_power = self.Rdroite[colonne]
					else:
						green_power = self.vegetation_map[colonne, ligne + 1] # Case au dessus
					correction  = power*log_factor(green_power)
					if tirage < self.alphaSouthNorth*self.p1*correction:
						if ligne == self.yp - 1: # Ce cas est traité par le processus de droite
							self.listeD.append([C, R + 1])
						else:
							self.fire_map[colonne, ligne + 1] = np.uint8(255)
							next_front   [(C, R + 1)] = np.uint8(255)

				if R > 0:
					tirage	= pseudo_random( self.glob_index(lexico_coord)*13427+self.time_step, self.time_step)
					if ligne == 0: # Ca implique que self.b > 0
						green_power = self.Rgauche[colonne]
					else:
						green_power = self.vegetation_map[colonne, ligne - 1] # Case au dessous
					correction  = power*log_factor(green_power)
					if tirage < self.alphaNorthSouth*self.p1*correction:
						if ligne == 0: # Ce cas est traité par le processus de gauche
							self.listeG.append([C, R - 1])
						else:
							self.fire_map[colonne, ligne - 1] = np.uint8(255)
							next_front   [(C, R - 1)] = np.uint8(255)

				if C < self.geometry-1:
					tirage	= pseudo_random( self.glob_index(lexico_coord)+self.time_step*42569, self.time_step)
					if colonne == self.xp - 1: #Ca implique que self.a < self.i - 1
						green_power = self.Rbas[ligne]
					else:
						green_power = self.vegetation_map[colonne + 1, ligne] # Case à droite
					correction  = power*log_factor(green_power)
					if tirage < self.alphaEastWest*self.p1*correction:
						if colonne == self.xp - 1: # Ce cas est traité par le processus d'en bas
							self.listeB.append([C + 1, R])
						else:
							self.fire_map[colonne + 1, ligne] = np.uint8(255)
							next_front   [(C + 1, R)] = np.uint8(255)

				if C > 0:
					tirage	= pseudo_random( self.glob_index(lexico_coord)*13427+self.time_step*42569, self.time_step)
					if colonne == 0: # Ca implique que self.a > 0
						green_power = self.Rhaut[ligne]
					else:
						green_power = self.vegetation_map[colonne - 1, ligne] # Case à gauche
					correction  = power*log_factor(green_power)
					if tirage < self.alphaWestEast*self.p1*correction:
						if colonne == 0: # Ce cas est traité par le processus d'en haut
							self.listeH.append([C - 1, R])
						else:
							self.fire_map[colonne - 1, ligne] = np.uint8(255)
							next_front   [(C - 1, R)] = np.uint8(255)

				# Si le feu est à son max,
				if fire == 255:
					# On regarde si il commence à faiblir pour s'éteindre au bout d'un moment :
					tirage = pseudo_random( self.glob_index(lexico_coord) * 52513 + self.time_step, self.time_step)
					if tirage < self.p2:
						self.fire_map[colonne, ligne] >>= 1
						next_front   [(C, R)] >>= 1
				else:
					# Foyer en train de s'éteindre.
					self.fire_map[colonne, ligne] >>= 1
					next_front   [(C, R)] >>= 1
					if next_front[(C, R)] == 0:
						next_front.pop((C, R))
			
			# On communique avec les autres processus
			self.echange_pendant_update()
			if self.b > 0:
				for c, r in self.Rgauche:
					next_front[(c, r)] = np.uint8(255)
			if self.b < self.j - 1:
				for c, r in self.Rdroite:
					next_front[(c, r)] = np.uint8(255)
			if self.a > 0:
				for c, r in self.Rhaut:
					next_front[(c, r)] = np.uint8(255)
			if self.a < self.i - 1:
				for c, r in self.Rbas:
					next_front[(c, r)] = np.uint8(255)
			
			# print(f"Je suis le processus {self.rank} et mon dictionnaire est {next_front}.")
			
			# A chaque itération, la végétation à l'endroit d'un foyer diminue
			self.fire_front = next_front
			for lexico_coord, _ in self.fire_front.items():
				C = lexico_coord[COLUMN]
				R = lexico_coord[ROW]
				colonne = C - self.a*self.xp
				ligne = R - self.b*self.yp
				if self.vegetation_map[colonne, ligne] > 0:
					self.vegetation_map[colonne, ligne] -= 1
			self.time_step += 1

			# On communique avec le processus 0
			self.echange_apres_update()
			taille = len(self.fire_front)
			taille = self.comm.reduce(taille)
			return None
		
		else: # Porcessus 0
			self.time_step += 1
			self.echange_apres_update()
			taille = 0
			taille = self.comm.reduce(taille)
			return taille > 0