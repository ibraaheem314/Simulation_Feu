
import pygame  as pg
import numpy   as np

class DisplayFire:
    """
    Permet l'affichage de la carte incendie et de la végétation en code couleur
    avec dans la couleur une composante verte pour la densité de végétation (entre 0 et 255)
    et une composante rouge pour l'incendie (avec une puissance comprise entre 0 et 255)
    Une cellule de discrétisation va correspondre un pixel dans la fenêtre d'affichage
    """
    def __init__( self, nb_cells_per_direction : int ):
        self.width = nb_cells_per_direction
        self.height= nb_cells_per_direction
        self.screen = pg.display.set_mode((self.width,self.height))
        self.pixels = None

    def update( self, fire_map : np.ndarray, vegetation_map : np.ndarray ):
        if self.pixels is None :
            self.pixels = pg.surfarray.pixels2d(self.screen)
        color_map = np.empty(shape=(self.width, self.height), dtype=np.uint32)
        color_map[:,:] = 65536*np.int32(fire_map[:,:]) + 256 * np.int32(vegetation_map[:,:])
        self.pixels[:,:] = color_map[:,:]
        pg.display.update()
