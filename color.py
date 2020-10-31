# A set of functions to be able to change the color of the text in the programs output
# Just used to add a little style to the program
GREEN = '\033[92m'
RESET = '\033[0m'
RED = '\033[91m'
BLUE = '\033[94m'


def colorText(color, text):
    return color + text + RESET


def green_string(text):
    return colorText(GREEN, text)


def blue_string(text):
    return colorText(BLUE, text)


def redString(text):
    return colorText(RED, text)
