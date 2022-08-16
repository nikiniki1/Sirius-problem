import numpy as np


class Domain:
    def __init__(self, xs, xe, nx, ys, ye, ny, lx, ly):
        self.xs = xs
        self.xe = xe
        self.nx = nx
        self.lx = lx

        self.dx = (xe - xs) / nx
        self.x = xs + np.arange(0, nx + 1) * self.dx

        self.ys = ys
        self.ye = ye
        self.ny = ny
        self.ly = ly

        self.dy = (ye - ys) / ny
        self.y = ys + np.arange(0, ny + 1) * self.dy

        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def __str__(self):
        return f"""
              Domain object:
                  2D domain: {[self.xs,self.xe]}x{[self.ys,self.ye]}
                  grid nx, ny: {self.nx, self.ny}
                  grid dx, dy: {self.dx, self.dy}
              """