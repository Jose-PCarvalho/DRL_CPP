import pygame
import numpy as np
from src.Environment.Grid import GridMap
from src.Environment.State import Position


class Vizualization:
    def __init__(self):
        self.window = None
        self.clock = None
        self.window_size = 512

    def render(self, mapa: GridMap, pos: Position):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        size = max(mapa.width * 2 - 1, mapa.height * 2 - 1)
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
                self.window_size / size
        )  # The size of a single grid square in pixels

        # First we draw the target
        a = mapa.center_map(pos.get_position())
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                print(a[:, i, j])
                if a[0, i, j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif a[1, i, j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (255, 255, 255),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif a[2, i, j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 0),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
                elif a[3, i, j] == 1:
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255),
                        pygame.Rect(
                            pix_square_size * np.array([j, i]),
                            (pix_square_size, pix_square_size),
                        ),
                    )

        # Now we draw the agent

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.flip(np.array([4, 4])) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(a.shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            for y in range(a.shape[2] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * y, 0),
                    (pix_square_size * y, self.window_size),
                    width=3,
                )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(4)

    def render_center(self, a_):

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        a=graph_to_RGB_array(a_)
        size = a.shape[1]
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
                self.window_size / size
        )  # The size of a single grid square in pixels

        # First we draw the target
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                pygame.draw.rect(
                    canvas,
                    a[:, i, j],
                    pygame.Rect(
                        pix_square_size * np.array([j, i]),
                        (pix_square_size, pix_square_size),
                    ),
                )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.flip(np.array([a.shape[1] / 2, a.shape[1] / 2]))) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(a.shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            for x in range(a.shape[1] + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, self.window_size),
                    width=3,
                )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(4)


def graph_to_RGB_array(a):
    rgb = np.zeros((3,a.shape[1], a.shape[2]), dtype=np.uint8)

    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            if a[0, i, j] == 1:
                rgb[0, i, j] = 255
            elif a[1, i, j] == 1:
                rgb[:, i, j] = [255, 255, 255]
            elif a[2, i, j] == 1:
                pass
            elif a[3, i, j] == 1:
                rgb[2, i, j] = 255
    return rgb
