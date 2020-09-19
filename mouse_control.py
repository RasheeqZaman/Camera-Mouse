import pyautogui
import sys



class MouseControl:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.screen_width = self.screen_width
        self.screen_height = self.screen_height
        self.min_range = 0.1
        self.max_range = 0.9

    def move_to(self, x, y):
        try:
            x = min(max((x-self.min_range)/(self.max_range-self.min_range), 0), 1)
            y = min(max((y-self.min_range)/(self.max_range-self.min_range), 0), 1)
            posx = int(self.screen_width*x)
            posy = int(self.screen_height*y)
            pyautogui.moveTo(posx, posy, duration = 0.1)
        except KeyboardInterrupt:
            sys.exit()