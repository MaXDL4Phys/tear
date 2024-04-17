class TextManager:
    """
    TextManager class for managing text properties and rendering.

    :param text: The text to be displayed.
    :type text: str
    :param font: The font family of the text.
    :type font: str
    :param color: The color of the text.
    :type color: str
    :param size: The size of the text.
    :type size: int
    :param position: The position of the text on the screen.
    :type position: tuple
    :param center: Specifies whether the text should be centered or not.
    :type center: bool

    :ivar text: The text to be displayed.
    :vartype text: str
    :ivar font: The font family of the text.
    :vartype font: str
    :ivar color: The color of the text.
    :vartype color: str
    :ivar size: The size of the text.
    :vartype size: int
    :ivar position: The position of the text on the screen.
    :vartype position: tuple
    :ivar center: Specifies whether the text should be centered or not.
    :vartype center: bool
    """
    def __init__(self, text, font, color, size, position, center):
        self.text = text
        self.font = font
        self.color = color
        self.size = size
        self.position = position
        self.center = center